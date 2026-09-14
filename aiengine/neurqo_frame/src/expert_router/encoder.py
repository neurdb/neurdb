# Standard library imports
import copy
from parser import sql_parser, table_parser
from parser.table_parser import *

# Local/project imports
from db.pg_conn import PostgresConnector
from utils import sql_utils


class Expr:
    def __init__(self, expr, list_kind=0):
        self.expr = expr
        self.list_kind = list_kind
        self.isInt = False
        self.val = 0

    def isCol(
        self,
    ):
        return isinstance(self.expr, dict) and "ColumnRef" in self.expr

    def getValue(self, value_expr):
        if "A_Const" in value_expr:
            value = value_expr["A_Const"]["val"]
            if "String" in value:
                return "'" + value["String"]["str"].replace("'", "''") + "'"
            elif "Integer" in value:
                self.isInt = True
                self.val = value["Integer"]["ival"]
                return str(value["Integer"]["ival"])
            else:
                raise "unknown Value in Expr"
        elif "TypeCast" in value_expr:
            if len(value_expr["TypeCast"]["typeName"]["TypeName"]["names"]) == 1:
                return (
                    value_expr["TypeCast"]["typeName"]["TypeName"]["names"][0][
                        "String"
                    ]["str"]
                    + " '"
                    + value_expr["TypeCast"]["arg"]["A_Const"]["val"]["String"]["str"]
                    + "'"
                )
            else:
                try:
                    # Case: WHERE (...) INTERVAL '1' year
                    if (
                        value_expr["TypeCast"]["typeName"]["TypeName"]["typmods"][0][
                            "A_Const"
                        ]["val"]["Integer"]["ival"]
                        == 2
                    ):
                        return (
                            value_expr["TypeCast"]["typeName"]["TypeName"]["names"][1][
                                "String"
                            ]["str"]
                            + " '"
                            + value_expr["TypeCast"]["arg"]["A_Const"]["val"]["String"][
                                "str"
                            ]
                            + "' month"
                        )
                    else:
                        return (
                            value_expr["TypeCast"]["typeName"]["TypeName"]["names"][1][
                                "String"
                            ]["str"]
                            + " '"
                            + value_expr["TypeCast"]["arg"]["A_Const"]["val"]["String"][
                                "str"
                            ]
                            + "' year"
                        )
                except KeyError:
                    # Case: WHERE (...) '1 year'::interval
                    # --> Breaks the above condition, as the 'typmods' key is not present
                    #
                    # 'interval'
                    casted_type = value_expr["TypeCast"]["typeName"]["TypeName"][
                        "names"
                    ][1]["String"]["str"]
                    # value: '1 year'
                    string_value = value_expr["TypeCast"]["arg"]["A_Const"]["val"][
                        "String"
                    ]["str"]
                    return f"'{string_value}'::{casted_type}"

        else:
            print(value_expr.keys())
            raise "unknown Value in Expr"

    def getAliasName(
        self,
    ):
        return self.expr["ColumnRef"]["fields"][0]["String"]["str"]

    def getColumnName(
        self,
    ):
        return self.expr["ColumnRef"]["fields"][1]["String"]["str"]

    def __str__(
        self,
    ):
        if self.isCol():
            return self.getAliasName() + "." + self.getColumnName()
        elif isinstance(self.expr, dict) and "A_Const" in self.expr:
            return self.getValue(self.expr)
        elif isinstance(self.expr, dict) and "TypeCast" in self.expr:
            return self.getValue(self.expr)
        elif isinstance(self.expr, list):
            if self.list_kind == 6:
                return "(" + ",\n".join([self.getValue(x) for x in self.expr]) + ")"
            elif self.list_kind == 10:
                return " AND ".join([self.getValue(x) for x in self.expr])
            else:
                raise "list kind error"

        else:
            raise "No Known type of Expr"


class SpecialSTACKPlusExpr(Expr):
    def isCol(
        self,
    ):
        return True

    def getValue(self, value_expr=None):
        alias = self.getAliasName()
        column = self.getColumnName()

        interval_size = self.expr["A_Expr"]["rexpr"]["TypeCast"]["arg"]["A_Const"][
            "val"
        ]["String"]["str"]
        return f"{alias}.{column} + '{interval_size}'::interval"

    def getAliasName(
        self,
    ):
        return self.expr["A_Expr"]["lexpr"]["ColumnRef"]["fields"][0]["String"]["str"]

    def getColumnName(
        self,
    ):
        return self.expr["A_Expr"]["lexpr"]["ColumnRef"]["fields"][1]["String"]["str"]

    def __str__(
        self,
    ):
        return self.getValue()


class TargetTable:
    def __init__(self, target):
        """
        {'location': 7, 'name': 'alternative_name', 'val': {'FuncCall': {'funcname': [{'String': {'str': 'min'}}], 'args': [{'ColumnRef': {'fields': [{'String': {'str': 'an'}}, {'String': {'str': 'name'}}], 'location': 11}}], 'location': 7}}}
        """
        self.target = target

    #         print(self.target)

    def getValue(
        self,
    ):
        columnRef = self.target["val"]["FuncCall"]["args"][0]["ColumnRef"]["fields"]
        return columnRef[0]["String"]["str"] + "." + columnRef[1]["String"]["str"]

    def __str__(
        self,
    ):
        try:
            return (
                self.target["val"]["FuncCall"]["funcname"][0]["String"]["str"]
                + "("
                + self.getValue()
                + ")"
                + " AS "
                + self.target["name"]
            )
        except:
            if "FuncCall" in self.target["val"]:
                return "count(*)"
            else:
                return "*"


class FromTable:
    def __init__(self, from_table):
        """
        {'alias': {'Alias': {'aliasname': 'an'}}, 'location': 168, 'inhOpt': 2, 'relpersistence': 'p', 'relname': 'aka_name'}
        """
        self.from_table = from_table
        if not "alias" in self.from_table:
            self.from_table["alias"] = {"Alias": {"aliasname": from_table["relname"]}}

    def getFullName(
        self,
    ):
        return self.from_table["relname"]

    def getAliasName(
        self,
    ):
        return self.from_table["alias"]["Alias"]["aliasname"]

    def __str__(
        self,
    ):
        try:
            return self.getFullName() + " AS " + self.getAliasName()
        except:
            print(self.from_table)
            raise


class Comparison:
    def __init__(self, comparison):
        self.comparison = comparison
        self.column_list = []
        if "A_Expr" in self.comparison:
            self.lexpr = Expr(comparison["A_Expr"]["lexpr"])
            self.column = str(self.lexpr)
            self.kind = comparison["A_Expr"]["kind"]
            if not "A_Expr" in comparison["A_Expr"]["rexpr"]:
                self.rexpr = Expr(comparison["A_Expr"]["rexpr"], self.kind)
            elif (
                comparison["A_Expr"]["rexpr"]["A_Expr"]["name"][0]["String"]["str"]
                == "+"
            ):
                self.rexpr = SpecialSTACKPlusExpr(
                    comparison["A_Expr"]["rexpr"], self.kind
                )

            else:
                self.rexpr = Comparison(comparison["A_Expr"]["rexpr"])

            self.aliasname_list = []

            if self.lexpr.isCol():
                self.aliasname_list.append(self.lexpr.getAliasName())
                self.column_list.append(self.lexpr.getColumnName())

            if self.rexpr.isCol():
                self.aliasname_list.append(self.rexpr.getAliasName())
                self.column_list.append(self.rexpr.getColumnName())

            self.comp_kind = 0
        elif "NullTest" in self.comparison:
            self.lexpr = Expr(comparison["NullTest"]["arg"])
            self.column = str(self.lexpr)
            self.kind = comparison["NullTest"]["nulltesttype"]

            self.aliasname_list = []

            if self.lexpr.isCol():
                self.aliasname_list.append(self.lexpr.getAliasName())
                self.column_list.append(self.lexpr.getColumnName())
            self.comp_kind = 1
        else:
            #             "boolop"
            self.kind = comparison["BoolExpr"]["boolop"]
            self.comp_list = [Comparison(x) for x in comparison["BoolExpr"]["args"]]
            self.aliasname_list = []
            for comp in self.comp_list:
                if comp.lexpr.isCol():
                    self.aliasname_list.append(comp.lexpr.getAliasName())
                    self.lexpr = comp.lexpr
                    self.column = str(self.lexpr)
                    self.column_list.append(comp.lexpr.getColumnName())
                    break
            self.comp_kind = 2

    def isCol(
        self,
    ):
        return False

    def __str__(
        self,
    ):

        if self.comp_kind == 0:
            Op = ""
            if self.kind == 0:
                Op = self.comparison["A_Expr"]["name"][0]["String"]["str"]
            elif self.kind == 7:
                if self.comparison["A_Expr"]["name"][0]["String"]["str"] == "!~~":
                    Op = "not like"
                else:
                    Op = "like"
            elif self.kind == 8:
                if self.comparison["A_Expr"]["name"][0]["String"]["str"] == "~~*":
                    Op = "ilike"
                else:
                    raise
            elif self.kind == 6:
                Op = "IN"
            elif self.kind == 10:
                Op = "BETWEEN"
            else:
                import json

                print(json.dumps(self.comparison, sort_keys=True, indent=4))
                raise "Operation ERROR"
            return str(self.lexpr) + " " + Op + " " + str(self.rexpr)
        elif self.comp_kind == 1:
            if self.kind == 1:
                return str(self.lexpr) + " IS NOT NULL"
            else:
                return str(self.lexpr) + " IS NULL"
        else:
            res = ""
            for comp in self.comp_list:
                if res == "":
                    res += "( " + str(comp)
                else:
                    if self.kind == 1:
                        res += " OR "
                    else:
                        res += " AND "
                    res += str(comp)
            res += ")"
            return res


class Table:
    def __init__(self, table_tree):
        self.name = table_tree["relation"]["RangeVar"]["relname"]
        self.column2idx = {}
        self.idx2column = {}
        self.column2type = {}
        for idx, columndef in enumerate(table_tree["tableElts"]):
            self.column2idx[columndef["ColumnDef"]["colname"]] = idx
            self.column2type[columndef["ColumnDef"]["colname"]] = columndef[
                "ColumnDef"
            ]["typeName"]["TypeName"]["names"][-1]["String"]["str"]
            # print(columndef["ColumnDef"]["typeName"]['TypeName']['names'],self.column2type[columndef["ColumnDef"]["colname"]],self.column2type[columndef["ColumnDef"]["colname"]] in ['int4','text','varchar'])
            assert self.column2type[columndef["ColumnDef"]["colname"]] in [
                "int4",
                "text",
                "varchar",
            ]
            if self.column2type[columndef["ColumnDef"]["colname"]] == "int4":
                self.column2type[columndef["ColumnDef"]["colname"]] = "int"
            else:
                self.column2type[columndef["ColumnDef"]["colname"]] = "str"
            self.idx2column[idx] = columndef["ColumnDef"]["colname"]

    def oneHotAll(self):
        return np.zeros((1, len(self.column2idx)))


class DB:
    def __init__(self, schema, TREE_NUM_IN_NET=40):
        from psqlparse import parse_dict

        parse_tree = parse_dict(schema)

        self.tables = []
        self.name2idx = {}
        self.table_names = []
        self.name2table = {}
        self.size = 0
        self.TREE_NUM_IN_NET = TREE_NUM_IN_NET
        for idx, table_tree in enumerate(parse_tree):
            self.tables.append(Table(table_tree["CreateStmt"]))
            self.table_names.append(self.tables[-1].name)
            self.name2idx[self.tables[-1].name] = idx
            self.name2table[self.tables[-1].name] = self.tables[-1]

        self.columns_total = 0

        for table in self.tables:
            self.columns_total += len(table.idx2column)

        self.size = len(self.table_names)

    def is_str(self, table, column):
        table = self.name2table[table]
        return table.column2type[column] == "str"

    def __len__(
        self,
    ):
        if self.size == 0:
            self.size = len(self.table_names)
        return self.size

    def oneHotAll(
        self,
    ):
        return np.zeros((1, self.size))

    def network_size(
        self,
    ):
        return self.TREE_NUM_IN_NET * self.size


class Sql2VecEmbeddingV2:
    def __init__(self, db_profile_res: DBTableInfo, checkpoint_file: str):
        self.pg_runner = None
        self.checkpoint_file = checkpoint_file
        self.query_encodings = self._load_checkpoint()
        self.db_profile_res = db_profile_res

        # Normalize table sizes
        self.table_sizes = np.log1p(np.array(self.db_profile_res.table_size_list))
        self.table_sizes = (self.table_sizes - self.table_sizes.min()) / (
            self.table_sizes.max() - self.table_sizes.min()
        )

    def encode_query(self, query_id, sql: str, train_database: str, pg_runner=None):
        """Encode a SQL query into join conditions, filter conditions, and table sizes."""
        if query_id in self.query_encodings:
            # Return a deep copy of the dictionary and its NumPy arrays
            return copy.deepcopy(self.query_encodings[query_id])

        # Initialize database runner
        if not pg_runner:
            self.pg_runner = PostgresConnector(train_database)
        else:
            self.pg_runner = pg_runner
        from psqlparse import parse_dict

        # Parse SQL query
        parse_result = parse_dict(sql)[0]["SelectStmt"]
        from_tables = [
            FromTable(from_clause["RangeVar"])
            for from_clause in parse_result["fromClause"]
        ]

        # Extract table information
        table_ids = []
        table_alias_fullname = {}
        for table in from_tables:
            alias = table.getAliasName()
            fullname = str(table)
            table_ids.append(self.db_profile_res.table_no_map[alias])
            table_alias_fullname[alias] = fullname

        # Parse where clause comparisons
        comparisons = [
            Comparison(comp) for comp in parse_result["whereClause"]["BoolExpr"]["args"]
        ]

        join_conditions = []
        filter_conditions = []

        for comp in comparisons:
            if len(comp.aliasname_list) == 2:
                # Join condition (e.g., A.id = B.a_id)
                alias1, alias2 = comp.aliasname_list
                table1_id = self.db_profile_res.table_no_map[alias1]
                table2_id = self.db_profile_res.table_no_map[alias2]
                col1 = comp.column_list[0]
                col2 = comp.column_list[1]
                col1_id = self.db_profile_res.attr_no_map_list[table1_id][col1]
                col2_id = self.db_profile_res.attr_no_map_list[table2_id][col2]
                # Default selectivity of 1.0 for joins unless filtered
                join_conditions.append([table1_id, col1_id, table2_id, col2_id])
            elif len(comp.aliasname_list) == 1:
                # Filter condition (e.g., A.name = 'foo')
                alias = comp.aliasname_list[0]
                table_id = self.db_profile_res.table_no_map[alias]
                col = comp.column_list[0]
                col_id = self.db_profile_res.attr_no_map_list[table_id][col]
                table_str = table_alias_fullname[alias]
                comp_str = str(comp)
                sel = self.pg_runner.get_selectivity(table_str, comp_str)
                filter_conditions.append([table_id, col_id, sel])

        # Create table mask and sizes
        table_mask = np.zeros(len(self.db_profile_res.table_size_list))
        table_mask[table_ids] = 1
        table_sizes = self.table_sizes * table_mask

        # Store encoded data
        encoded = {
            "join_conditions": np.array(join_conditions, dtype=np.float32),
            "filter_conditions": np.array(filter_conditions, dtype=np.float32),
            "table_sizes": table_sizes.astype(np.float32),
        }

        self.query_encodings[query_id] = encoded
        self._save_checkpoint()
        return encoded

    def _load_checkpoint(self):
        """Load encoded queries from checkpoint file."""
        if os.path.exists(self.checkpoint_file):
            print(f"Loading from the checkpoint {self.checkpoint_file}")
            with open(self.checkpoint_file, "r") as f:
                data = json.load(f)
                return {
                    query_id: {
                        "join_conditions": np.array(
                            enc["join_conditions"], dtype=np.float32
                        ),
                        "filter_conditions": np.array(
                            enc["filter_conditions"], dtype=np.float32
                        ),
                        "table_sizes": np.array(enc["table_sizes"], dtype=np.float32),
                    }
                    for query_id, enc in data.items()
                }
        else:
            print(f"Cannot find the file from the checkpoint {self.checkpoint_file}")
        return {}

    def _save_checkpoint(self):
        """Save encoded queries to checkpoint file."""
        serializable_data = {
            query_id: {
                "join_conditions": enc["join_conditions"].tolist(),
                "filter_conditions": enc["filter_conditions"].tolist(),
                "table_sizes": enc["table_sizes"].tolist(),
            }
            for query_id, enc in self.query_encodings.items()
        }
        with open(self.checkpoint_file, "w") as f:
            json.dump(serializable_data, f, indent=2)


class QueryFeatureBuilder:
    def __init__(
        self, table_info: table_parser.DBTableInfo, possible_join_attrs: np.ndarray
    ):

        self.table_info = table_info
        self.n_tables = len(table_info.attr_ranges_list)
        self.maxn_attrs_single_table = max(
            table_columns.shape[0] for table_columns in table_info.attr_ranges_list
        )

        # min max for each column
        self.attr_ranges_all = np.concatenate(
            table_info.attr_ranges_list, axis=0, dtype=np.float64
        )
        self.attr_types_all = np.concatenate(
            table_info.attr_no_types_list, dtype=np.int64
        )

        self.n_attrs_total = self.attr_types_all.shape[0]

        # for feature calc only
        self.attr_lbds = np.zeros_like(
            self.attr_ranges_all, dtype=self.attr_ranges_all.dtype
        )
        self.attr_lbds[:, 0] = self.attr_ranges_all[:, 0]
        self.attr_lbds[:, 1] = self.attr_ranges_all[:, 0]
        # flatten 2D into 1D
        self.attr_lbds = np.reshape(self.attr_lbds, [-1])

        # used for the min-max normalization that rescales the values into the range [−1, 1]
        self.attr_norm_base = np.zeros_like(
            self.attr_ranges_all, dtype=self.attr_ranges_all.dtype
        )
        attr_range_differ = self.attr_ranges_all[:, 1] - self.attr_ranges_all[:, 0]
        self.attr_norm_base[:, 0] = attr_range_differ
        self.attr_norm_base[:, 1] = attr_range_differ
        # flatten 2D into 1D
        self.attr_norm_base = np.reshape(self.attr_norm_base, [-1])
        # Add a small value to any zero entries to prevent division by zero
        self.attr_norm_base = np.where(
            self.attr_norm_base == 0, self.attr_norm_base + 1e-8, self.attr_norm_base
        )

        # this is got from the base.sql not a specific workload
        self.join_id_no_map, self.n_possible_joins = self._build_join_id_no_map(
            possible_join_attrs
        )

    def encode_query(self, sql: str) -> np.ndarray:
        """
        Featurize one read sql.
        """
        sql = sql.lower().strip()
        if not sql_utils.is_read_sql(sql):
            raise
        # init the features result
        feature = np.zeros(
            self.n_tables + self.n_possible_joins + self.n_attrs_total * 2,
            dtype=np.float64,
        )

        # parser sql and fill the feature
        join_conds, relevant_tables, equi_classes, attr_range_conds, join_type = (
            sql_parser.parse_one_sql(sql, self.table_info)
        )

        feature[relevant_tables] = 1
        if join_conds is not None:
            # Encode conjunctive join conds
            join_idx_in_feature = self.calc_join_nos(join_conds)
            join_idx_in_feature += (
                self.n_tables
            )  # Shift the join position in feature vector
            feature[join_idx_in_feature] = 1

        cursor = self.n_tables + self.n_possible_joins
        attr_range_conds_array = np.array(attr_range_conds, dtype=np.float64)
        feature[cursor : cursor + self.n_attrs_total * 2] = (
            (attr_range_conds_array - self.attr_lbds) / self.attr_norm_base
        ) * 2.0 - 1

        return feature

    def _build_join_id_no_map(self, possible_join_attrs: np.ndarray):
        # assign the join id for each join in base query set.
        total_join_patterns = possible_join_attrs.shape[0]
        join_attrs_trans = possible_join_attrs.transpose()

        # It uses table_id * max_attr_num + attr_id to create a unique identifier for each attribute.
        table1, t1_attr, table2, t2_attr = (
            join_attrs_trans[0],
            join_attrs_trans[1],
            join_attrs_trans[2],
            join_attrs_trans[3],
        )
        table_attr_id_1 = table1 * self.maxn_attrs_single_table + t1_attr
        table_attr_id_2 = table2 * self.maxn_attrs_single_table + t2_attr
        # total number of possible attributes in all tables, table.attribute
        total_table_attribute_num = self.maxn_attrs_single_table * self.n_tables

        # assign each join operation a unique id, and mapper
        join_ids = set()
        equi_relations = {}
        for i in range(total_join_patterns):
            id_1 = table_attr_id_1[i]
            id_2 = table_attr_id_2[i]
            if id_1 <= id_2:
                join_id = id_1 * total_table_attribute_num + id_2
                symm_join_id = id_2 * total_table_attribute_num + id_1
            else:
                join_id = id_2 * total_table_attribute_num + id_1
                symm_join_id = id_1 * total_table_attribute_num + id_2
            equi_relations[join_id] = symm_join_id
            join_ids.add(join_id)
        join_ids = list(join_ids)
        join_ids.sort()

        # both symm_join and join has same index, i
        join_id_no_map = {}
        for i, join_id in enumerate(join_ids):
            join_id_no_map[join_id] = i
            symm_join_id = equi_relations[join_id]
            join_id_no_map[symm_join_id] = i

        n_possible_joins = len(join_id_no_map)
        return join_id_no_map, n_possible_joins

    def calc_join_nos(self, join_conds: List[List[int]]) -> np.ndarray:
        """"""
        # join_no: a number from [0, self.n_possible_joins)
        join_idxes = self._calc_join_ids(join_conds)
        for i in range(join_idxes.shape[0]):
            join_idxes[i] = self.join_id_no_map[join_idxes[i]]
        return join_idxes

    def _calc_join_ids(self, join_conds: List[List[int]]) -> np.ndarray:
        """
        # join_id: a number from [0, M * M), where M = self.maxn_attrs_single_table * self.n_tables
        """
        join_conds_trans = np.transpose(join_conds)
        m1 = join_conds_trans[0] * self.maxn_attrs_single_table + join_conds_trans[1]
        m2 = join_conds_trans[2] * self.maxn_attrs_single_table + join_conds_trans[3]
        M = self.maxn_attrs_single_table * self.n_tables
        return m1 * M + m2
