# Standard library imports
import re

# Local/project imports
from parser import table_parser
from typing import Dict, List


def parse_filter(table_info: table_parser.DBTableInfo, plan: Dict) -> List[str]:
    # Helper function to extract and validate conditions based on table_info
    def extract_conditions(alias_table: str, filter_str: str) -> List[str]:
        parsed_conditions = []

        # Split conditions based on logical operators (AND/OR)
        condition_pattern = r"\b(AND|OR)\b"
        conditions = re.split(condition_pattern, filter_str)

        # Iterate through each condition
        for cond in conditions:
            cond = cond.strip().strip("()")
            if not cond or cond in ["AND", "OR"]:
                continue

            table_number = table_info.table_no_map[alias_table]
            table_names = table_info.no_table_map.get(table_number)
            table_columns = []
            for table_name in table_names:
                if table_name in table_info.table_attr_map:
                    table_columns.extend(table_info.table_attr_map[table_name])
            if len(table_columns) == 0:
                raise

            # Split condition at "=" to check for valid column name
            parts = cond.split("=")
            if len(parts) == 2:
                column_candidate = parts[0].strip()
                value = parts[1].strip()

                # parse the column name from the list
                sorted_column_names = sorted(table_columns, key=len, reverse=True)
                for column_name in sorted_column_names:
                    if column_name in column_candidate:
                        parsed_conditions.append(
                            f"{alias_table}.{column_name} = {value}"
                        )
                        break

        return parsed_conditions

    # Get the table alias
    alias_table = plan.get("Alias")
    if not alias_table:
        # Traverse to find alias in parent nodes if necessary
        pl = plan
        while "parent" in pl:
            pl = pl["parent"]
            alias_table = pl.get("Alias")
            if alias_table:
                break

    # Parse filter conditions if alias_table is found
    filter_predicates = []
    if alias_table:
        if "Filter" in plan:
            filter_predicates.extend(
                extract_conditions(alias_table, plan["Filter"][1:-1])
            )
        if "Index Cond" in plan and plan["Index Cond"][-2].isnumeric():
            filter_predicates.extend(
                extract_conditions(alias_table, plan["Index Cond"][1:-1])
            )
        if "Recheck Cond" in plan and plan["Recheck Cond"][-2].isnumeric():
            filter_predicates.extend(
                extract_conditions(alias_table, plan["Recheck Cond"][1:-1])
            )
    else:
        # Error if no alias found
        if "Filter" in plan or "Index Cond" in plan or "Recheck Cond" in plan:
            print(plan)
            raise ValueError("Alias not found")

    return filter_predicates


def parse_join(table_info: table_parser.DBTableInfo, json_node: Dict) -> (str, List):
    """
    Parse node and return string as t.id = mi_idx.movie_id
    """
    join_cond = None
    if "Hash Cond" in json_node:
        join_cond = json_node["Hash Cond"]
    elif "Join Filter" in json_node:
        join_cond = json_node["Join Filter"]
    elif "Index Cond" in json_node and not json_node["Index Cond"][-2].isnumeric():
        join_cond = json_node["Index Cond"]

    # consist the encoding as table.column
    unique_tables = []
    if join_cond is not None:
        twoCol = join_cond[1:-1].split(" = ")
        res = []
        for col in twoCol:
            # if no table name
            if len(col.split(".")) == 1:
                # Add alias, relation name, or index name as the last fallback (removing _pkey if necessary)
                if "Alias" in json_node:
                    table_prefix = json_node["Alias"]
                elif "Relation Name" in json_node:
                    table_prefix = json_node["Relation Name"]
                elif "Index Name" in json_node:
                    # for talbes like 'comments__pkey'
                    if "_pkey" in json_node["Index Name"]:
                        table_prefix = json_node["Index Name"].replace("_pkey", "")
                    else:
                        table_prefix = table_info.index_table_map[
                            json_node["Index Name"]
                        ]
                else:
                    raise
                res.append(f"{table_prefix}.{col}")
            else:
                res.append(col)

        join_cond = " = ".join(sorted(res))
        # Regular expression to match table aliases (before the dot)
        pattern = re.compile(r"(\w+)\.\w+")
        tables = pattern.findall(join_cond)
        unique_tables = list(set(tables))

    # check
    for table_name in unique_tables:
        assert table_name in table_info.table_no_map, table_name

    return join_cond, unique_tables
