# Standard library imports
import json
import re
import traceback

# Local/project imports
from parser.entiy import DBTableInfo, WorkloadQueryInfo
from typing import Any, Dict, List, Tuple

# Third-party imports
import networkx as nx
import numpy as np
import pglast
from pglast import ast
from pglast.visitors import Visitor
from scipy.cluster.hierarchy import DisjointSet
from utils import date_utils, file_utils

epsilon_for_float = 1e-6


class SQLInfoExtractor(Visitor):
    def __init__(self):
        super().__init__()
        self.tables = set()
        self.joins = set()
        self.predicates = set()
        self.aliasname_fullname = {}

    def visit(self, ancestors, node):

        # Extract table names and aliases
        if isinstance(node, ast.RangeVar):
            table_name = node.relname
            self.tables.add(table_name)
            if hasattr(node, "alias") and node.alias is not None:
                alias_name = node.alias.aliasname
                self.aliasname_fullname[alias_name] = table_name

        # Extract explicit join information
        if isinstance(node, ast.JoinExpr):
            if isinstance(node.larg, ast.RangeVar) and isinstance(
                node.rarg, ast.RangeVar
            ):
                left_table = node.larg.relname
                right_table = node.rarg.relname
                if node.larg.alias:
                    left_table = node.larg.alias.aliasname
                if node.rarg.alias:
                    right_table = node.rarg.alias.aliasname
                if node.quals and isinstance(node.quals, ast.A_Expr):
                    lexpr = (
                        node.quals.lexpr.fields
                        if hasattr(node.quals.lexpr, "fields")
                        else []
                    )
                    rexpr = (
                        node.quals.rexpr.fields
                        if hasattr(node.quals.rexpr, "fields")
                        else []
                    )
                    if len(lexpr) == 2 and len(rexpr) == 2:
                        self.joins.add(
                            (
                                (lexpr[0].sval, lexpr[1].sval),
                                (rexpr[0].sval, rexpr[1].sval),
                            )
                        )

        # Extract implicit join conditions from WHERE clause
        if isinstance(node, ast.A_Expr):
            if node.name[0].sval == "=":
                lexpr = node.lexpr.fields if hasattr(node.lexpr, "fields") else []
                rexpr = node.rexpr.fields if hasattr(node.rexpr, "fields") else []
                # for join, both have values
                if len(lexpr) == 2 and len(rexpr) == 2:
                    self.joins.add(
                        ((lexpr[0].sval, lexpr[1].sval), (rexpr[0].sval, rexpr[1].sval))
                    )
                elif len(lexpr) == 1 and len(rexpr) == 1:
                    self.joins.add((("", lexpr[0].sval), ("", rexpr[0].sval)))

            # Extract other predicates
            if node.name[0].sval in (">", "=", "<", "<=", ">="):
                lexpr = node.lexpr.fields if hasattr(node.lexpr, "fields") else []
                if hasattr(node.rexpr, "val"):
                    if hasattr(node.rexpr.val, "ival"):
                        rexpr = node.rexpr.val.ival
                    elif hasattr(node.rexpr.val, "sval"):
                        rexpr = node.rexpr.val.sval
                    else:
                        rexpr = None
                    if len(lexpr) == 2:
                        self.predicates.add(
                            ((lexpr[0].sval, lexpr[1].sval), node.name[0].sval, rexpr)
                        )
                    elif len(lexpr) == 1:
                        self.predicates.add(
                            (("", lexpr[0].sval), node.name[0].sval, rexpr)
                        )
                elif hasattr(node.rexpr, "arg"):
                    if hasattr(node.rexpr.arg, "val"):
                        if hasattr(node.rexpr.arg.val, "sval"):
                            rexpr = node.rexpr.arg.val.sval
                            if len(lexpr) == 2:
                                self.predicates.add(
                                    (
                                        (lexpr[0].sval, lexpr[1].sval),
                                        node.name[0].sval,
                                        rexpr,
                                    )
                                )
                            elif len(lexpr) == 1:
                                self.predicates.add(
                                    (("", lexpr[0].sval), node.name[0].sval, rexpr)
                                )

        # Handle subqueries and additional join conditions
        if isinstance(node, ast.SubLink):
            # Extract join conditions in subquery
            subselect = node.subselect
            if subselect:
                self(subselect)

        if isinstance(node, ast.BoolExpr):
            # Traverse all arguments in AND/OR expressions
            for arg in node.args:
                self(arg)


def parse_queries_on_batch(
    file_path: str, table_info: DBTableInfo
) -> WorkloadQueryInfo:
    # read queries from file
    # lines: list of query strings
    # table_no_map: {table_name: table_no}
    # attr_no_map_list: [{attr_name: attr_no}, {attr_name: attr_no}]
    # attr_no_types_list: [[attr_type]], [attr_type]]
    # attr_ranges_list: [ [[min,max], [min, max]], ..]

    # result map
    join_conds_list = []
    equi_classes_list = []
    filter_attr_range_conds_list = []
    join_type_list = []
    relevant_tables_list = []

    sql_lines = file_utils.read_all_lines(file_path)

    for line in sql_lines:
        # in some file, the label is added after the sql
        sql = line.strip()
        join_conds, relevant_tables, equi_classes, attr_range_conds, join_type = (
            parse_one_sql(sql, table_info)
        )

        equi_classes_list.append(equi_classes)
        join_conds_list.append(join_conds)
        filter_attr_range_conds_list.append(attr_range_conds)
        join_type_list.append(join_type)
        relevant_tables_list.append(relevant_tables)

    # Process each equivalence class
    possible_join_attrs = []
    for equi_classes in equi_classes_list:
        for equi_class in equi_classes.subsets():
            table_attr_list = sorted(list(equi_class))
            for i, l_table_attr in enumerate(table_attr_list):
                for j in range(i + 1, len(table_attr_list)):
                    r_table_attr = table_attr_list[j]
                    l_table_no, l_attr_no = get_table_no_and_attr_no(
                        l_table_attr,
                        table_info.table_no_map,
                        table_info.attr_no_map_list,
                    )
                    r_table_no, r_attr_no = get_table_no_and_attr_no(
                        r_table_attr,
                        table_info.table_no_map,
                        table_info.attr_no_map_list,
                    )
                    possible_join_attrs.append(
                        [l_table_no, l_attr_no, r_table_no, r_attr_no]
                    )
    possible_join_attrs = np.array(possible_join_attrs, dtype=np.int64)

    result = WorkloadQueryInfo(
        possible_join_attrs,
        join_conds_list,
        filter_attr_range_conds_list,
        join_type_list,
        relevant_tables_list,
    )

    return result


def simple_parse(sql: str) -> (Dict, List, List, List):
    node = pglast.parse_sql(sql)
    extractor = SQLInfoExtractor()
    extractor(node)

    short_full_table_name_map = extractor.aliasname_fullname
    join_conds = list(extractor.joins)
    filter_conds = list(extractor.predicates)
    used_tables = list(extractor.tables)
    return short_full_table_name_map, join_conds, filter_conds, used_tables


def parse_one_sql(_sql, table_info: DBTableInfo):
    """
    Encoding join and filter conditions in the SQL query
    JOIN: [table no, attr no, table2 no, attr2 no]
    FILTER: [attr1 range, attr2 range, ...]
    :param _sql: string, looks like select count * from A, B,, X where A.a1 = B.b2 and A.a2 = X.x1 and A.a3 >=1 and A.a3 < 10 and B.b2 = 1 and X.x1 <= 100
    :param table_info: DBTableInfo object
    :param table_info.table_no_map: table_name-table_id dict
    :param table_info.attr_no_map_list: list of dicts, each element is a attr_name-attr_no dict
    :param table_info.attr_no_type_map_list: list of dicts, each element is a attr_no-data_type dict
    :param table_info.attr_ranges_list: list of numpy ndarray. Each element r is a M x 2 matrix. r[i,0] and r[i,1] denotes the lower and upper bound of the ith attr
    :return:
    """
    sql = _sql.lower().strip()
    idx = sql.find("select")
    assert idx >= 0
    sql = sql[idx:]

    join_type = None

    # Parse the SQL query and get join predicates and filter predicates
    short_full_table_name_map, join_predicates, filter_predicates, used_tables = (
        simple_parse(sql)
    )
    # # update table_info
    # for short_name in short_full_table_name_map:
    #     full_name = short_full_table_name_map[short_name]
    #     table_info.table_no_map[short_name] = table_info.table_no_map[full_name]

    # 1. fill the relevant_tables
    relevant_tables = []
    for full_name in used_tables:
        table_no = table_info.table_no_map[full_name]
        relevant_tables.append(table_no)
    relevant_tables.sort()

    # 2. encode join conditions
    all_possible_join_conds = []
    equi_classes = _get_equi_classes(join_predicates)
    for equi_class in equi_classes.subsets():
        table_attr_list = list(equi_class)
        # table_attr_list.sort()
        for i, l_table_attr in enumerate(table_attr_list):
            for j, r_table_attr in enumerate(table_attr_list):
                if i != j:
                    l_table_no, l_attr_no = get_table_no_and_attr_no(
                        l_table_attr,
                        table_info.table_no_map,
                        table_info.attr_no_map_list,
                    )
                    r_table_no, r_attr_no = get_table_no_and_attr_no(
                        r_table_attr,
                        table_info.table_no_map,
                        table_info.attr_no_map_list,
                    )
                    all_possible_join_conds.append(
                        (l_table_no, l_attr_no, r_table_no, r_attr_no)
                    )
    all_possible_join_conds.sort()
    if len(all_possible_join_conds) == 0:
        join_conds = None
    else:
        join_conds = np.array(all_possible_join_conds, dtype=np.int64)

    # 3. encode filter conditions
    attr_range_conds = [x.copy() for x in table_info.attr_ranges_list]

    for filter_predicate in filter_predicates:
        (lhs, op, rhs) = filter_predicate
        lhs_table_no, lhs_attr_no = parse_predicate_lhs(
            lhs, table_info.table_no_map, table_info.attr_no_map_list
        )
        try:
            parsed_filter_value = parse_predicate_rhs(rhs)
        except Exception as e:
            print("filter_predicate =", filter_predicate, "sql =", sql)
            traceback.print_exc()
            raise Exception()

        # here if it's a string
        if parsed_filter_value is None:
            continue
        # 0 for int, 1 for float
        attr_type = table_info.attr_no_types_list[lhs_table_no][lhs_attr_no]
        if attr_type == 0:
            epsilon = 0.5
        else:
            epsilon = epsilon_for_float
        if op == "=":
            attr_range_conds[lhs_table_no][lhs_attr_no][0] = (
                parsed_filter_value - epsilon
            )
            attr_range_conds[lhs_table_no][lhs_attr_no][1] = (
                parsed_filter_value + epsilon
            )
        elif op == "<=":
            attr_range_conds[lhs_table_no][lhs_attr_no][1] = (
                parsed_filter_value + epsilon
            )
        elif op == "<":
            attr_range_conds[lhs_table_no][lhs_attr_no][1] = (
                parsed_filter_value - epsilon
            )
        elif op == ">=":
            attr_range_conds[lhs_table_no][lhs_attr_no][0] = (
                parsed_filter_value - epsilon
            )
        else:  # '>'
            attr_range_conds[lhs_table_no][lhs_attr_no][0] = (
                parsed_filter_value + epsilon
            )

    assert len(relevant_tables) > 0

    attr_range_conds = np.concatenate(attr_range_conds, axis=0, dtype=np.float64)
    attr_range_conds = np.reshape(attr_range_conds, [-1])
    assert join_conds is None or (join_conds.shape[0] > 0)

    return join_conds, relevant_tables, equi_classes, attr_range_conds, join_type


def parse_predicate_lhs(lhs: Tuple, table_no_map: Dict, attr_no_map_list: Dict):
    if len(lhs) == 2:
        table_name = lhs[0]
        attr = lhs[1].strip().strip("()")
        if table_name in table_no_map:
            table_no = table_no_map[table_name]
            attr_no_map = attr_no_map_list[table_no]
            assert attr in attr_no_map
            attr_no = attr_no_map[attr]
            return table_no, attr_no
    else:
        print("this is not table.col format, lhs =", lhs)
        raise Exception()


def parse_predicate_rhs(rhs: Any):
    # handle int or float
    try:
        return float(rhs)
    except ValueError:
        pass

    # handle the date format
    # todo: handle string later.
    val = date_utils.is_time_format(rhs)
    return val


def get_table_no_and_attr_no(
    table_attr: Tuple, table_no_map: Dict, attr_no_map_list: Dict
):
    table_name, attr_name = table_attr[0], table_attr[1]
    # if table_name not in table_no_map:
    #     raise f"{table_name} is not in {table_no_map}"
    table_no = table_no_map[table_name]
    attr_no_map = attr_no_map_list[table_no]
    if attr_name not in attr_no_map:
        raise
    attr_no = attr_no_map[attr_name]
    return table_no, attr_no


def _get_equi_classes(join_conds: List[Tuple]):
    equi_classes = DisjointSet()
    for join_cond in join_conds:
        lhs, rhs = join_cond
        equi_classes.add(lhs)
        equi_classes.add(rhs)
        equi_classes.merge(lhs, rhs)
    return equi_classes


# ------------------------ for read sqls ------------------------


def parse_value_from_update_sql(sql: str):
    # Regular expression to extract the values from the SET clause
    set_pattern = re.compile(r"set\s*\((.*?)\)\s*=\s*\((.*?)\)", re.IGNORECASE)
    # Regular expression to extract the values from the WHERE clause
    where_pattern = re.compile(r"where\s*(.*)", re.IGNORECASE)

    # Find the SET values
    set_match = set_pattern.search(sql)
    if set_match:
        # Extract the values inside the parentheses in the SET clause
        set_values = set_match.group(2).split(",")
        # Convert them to float
        set_values = [float(val) for val in set_values]
    else:
        raise ValueError("No valid SET clause found")

    # Find the WHERE clause
    where_match = where_pattern.search(sql)
    if where_match:
        # Extract the conditions after the WHERE keyword
        where_clause = where_match.group(1)
        # Extract the values from the WHERE clause using a regex that includes negative numbers
        where_values = re.findall(r"=\s*(-?\d+)", where_clause)
        # Convert them to float
        where_values = [float(val) for val in where_values]
    else:
        raise ValueError("No valid WHERE clause found")

    return where_values, set_values


def extract_delete_where_values(sql: str):
    # Regular expression to extract the WHERE clause
    where_pattern = re.compile(r"where\s*(.*)", re.IGNORECASE)

    # Find the WHERE clause
    where_match = where_pattern.search(sql)
    if where_match:
        # Extract the conditions after the WHERE keyword
        where_clause = where_match.group(1)
        # Extract the values from the WHERE clause using regular expressions
        where_values = re.findall(r"=\s*(-?\d+)", where_clause)
        # Convert them to float
        where_values = [float(val) for val in where_values]
    else:
        raise ValueError("No valid WHERE clause found")

    return where_values


def extract_insert_values(sql: str):
    terms = sql.split(" values")
    set_values = (terms[1].strip()[1:-2]).split(",")
    return set_values


def update_table_aliases_with_psqlparse(
    db_info_dic: Dict,
    queries: List[str],
    output_file: str = "./experiment/ori_table_info_stack.json",
) -> Dict:
    from psqlparse import parse_dict

    final_alias_map = {}

    # Process each SQL file in the directory
    for sql_content in queries:

        try:
            parse_result = parse_dict(sql_content)[0]["SelectStmt"]
        except Exception as e:
            raise e

        # Initialize the alias-to-table mapping
        alias_to_table = {}

        # Process FROM clause to get tables and aliases
        from_tables = parse_result.get("fromClause", [])

        for from_table in from_tables:
            table_info = from_table["RangeVar"]
            table_name = table_info["relname"]

            # Handle alias if present
            if "alias" in table_info:
                alias_name = table_info["alias"]["Alias"]["aliasname"]
                alias_to_table[alias_name] = table_name

        final_alias_map.update(alias_to_table)
    print(final_alias_map)

    final_alias_id_map = {}
    for table_name, table_id in db_info_dic["table_no_map"].items():
        for alias, full_name in final_alias_map.items():
            if table_name == full_name:
                final_alias_id_map[alias] = table_id

    print(final_alias_id_map)

    db_info_dic["table_no_map"].update(final_alias_id_map)

    # Write the updated dictionary to a JSON file
    with open(output_file, "w") as f:
        json.dump(db_info_dic, f, indent=4)

    print(f"Updated table info with aliases saved to {output_file}")
    return db_info_dic


def simple_parse_sql(sql: str):
    """Parses a SQL string into (nx.Graph, a list of join condition strings).
    e.g., join_conds = ['cn1.id = mc1.company_id', 'cn2.id = mc2.company_id']
    e.g., graph = {'mc1': {
                    'cn1': {0: {'join_keys': {'cn1': 'id', 'mc1': 'company_id'}}},
                    'mi_idx1': {0: {'join_keys': {'mc1': 'movie_id', 'mi_idx1': 'movie_id'}}},
                    't1': {0: {'join_keys': {'mc1': 'movie_id', 't1': 'id'}}}}}
    Both use aliases to refer to tables.
    """
    join_conds = _GetJoinConds(sql)
    graph = _GetGraph(join_conds)
    join_conds = [f"{t1}.{c1} = {t2}.{c2}" for t1, c1, t2, c2 in join_conds]
    return graph, join_conds


def _GetJoinConds(sql):
    """Returns a list of join conditions in the form of (t1, c1, t2, c2)."""
    join_cond_pat = re.compile(
        r"""
        (\w+)  # 1st table
        \.     # the dot "."
        (\w+)  # 1st table column
        \s*    # optional whitespace
        =      # the equal sign "="
        \s*    # optional whitespace
        (\w+)  # 2nd table
        \.     # the dot "."
        (\w+)  # 2nd table column
        """,
        re.VERBOSE,
    )
    join_conds = join_cond_pat.findall(sql)
    return _DedupJoinConds(join_conds)


def _GetGraph(join_conds):
    g = nx.MultiGraph()
    for t1, c1, t2, c2 in join_conds:
        g.add_edge(t1, t2, join_keys={t1: c1, t2: c2})
    return g


def _FormatJoinCond(tup):
    t1, c1, t2, c2 = tup
    return f"{t1}.{c1} = {t2}.{c2}"


def _DedupJoinConds(join_conds):
    """join_conds: list of 4-tuple (t1, c1, t2, c2)."""
    canonical_join_conds = [_CanonicalizeJoinCond(jc) for jc in join_conds]
    return sorted(set(canonical_join_conds))


def _CanonicalizeJoinCond(join_cond):
    """join_cond: 4-tuple"""
    t1, c1, t2, c2 = join_cond
    if t1 < t2:
        return join_cond
    return t2, c2, t1, c1
