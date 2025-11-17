# Standard library imports
import csv
import json
import os
import re
from collections import defaultdict

# Local/project imports
from parser.entiy import DBTableInfo
from typing import Dict, List, Tuple

# Third-party imports
import numpy as np
import pandas as pd
from utils import file_utils


def get_index_table(create_index_path: str) -> dict:
    lines = file_utils.read_all_lines(create_index_path)
    index_table = {}
    for _line in lines:
        line = _line.lower()
        if line.startswith("create index"):
            parts = line.split()
            index_name = parts[2]
            table_name = parts[4].split("(")[0]
            index_table[index_name] = table_name

    return index_table


def get_all_table_attr_infos(create_tables_path: str) -> (List, Dict, Dict):
    """
    Read a create_tables.sql file and return a list of table attribute information.
    table_attr_types_map: {name:type}
    """
    lines = file_utils.read_all_lines(create_tables_path)
    table_attr_infos_list = []
    table_attr_types_map = {}
    table_attr_map = {}
    for _line in lines:
        line = _line.lower()
        if line.startswith("create table"):
            table_name, attr_names, attr_type_list = _get_attr_infos_from_create_sql(
                line.strip()
            )
            table_attr_infos_list.append((table_name, attr_names, attr_type_list))
            table_attr_types_map[table_name] = attr_type_list
            table_attr_map[table_name] = attr_names

    table_names = list(table_attr_types_map.keys())
    table_names.sort()

    return table_names, table_attr_infos_list, table_attr_types_map, table_attr_map


def retrieve_db_info(
    table_names: List[str],
    table_attr_types_map: Dict,
    table_attr_map: Dict,
    index_table_map: Dict,
    data_dir: str,
    tables_info_path: str,
    all_sql_path: str,
) -> DBTableInfo:
    """
    Summarize and save the data info to tables_info_path
    """
    if not os.path.exists(tables_info_path):
        from parser import sql_parser

        print("tables_info not exist, summarizing....")
        table_info_json = _gen_db_info_json(
            table_names, table_attr_types_map, table_attr_map, index_table_map, data_dir
        )
        # update the alias names to the table path
        sql_lines = file_utils.read_all_lines(all_sql_path)
        for line in sql_lines:
            sql = line.strip()
            short_full_table_name_map, _, _, _ = sql_parser.simple_parse(sql)
            for short_name in short_full_table_name_map:
                full_name = short_full_table_name_map[short_name]
                table_info_json["table_no_map"][short_name] = table_info_json[
                    "table_no_map"
                ][full_name]
        print(f"tables_info is saved into {tables_info_path}")
        with open(tables_info_path, "w") as writer:

            json.dump(table_info_json, writer, indent=4)

    return load_db_info_json(tables_info_path)


def _get_attr_infos_from_create_sql(create_sql: str) -> (List, List, List):
    table_name, attr_descs, attr_extra_infos = _parse_create_sql(create_sql)
    attr_type_list = []
    attr_names = []
    for attr_desc in attr_descs:
        attr_name = attr_desc[0]
        attr_type = attr_desc[1]
        attr_names.append(attr_name)

        if attr_type in {"bigint", "integer", "smallint", "serial"}:
            attr_type_list.append(0)
        elif attr_type == "double":
            attr_type_list.append(1)
        else:
            attr_type_list.append(-1)

    return table_name, attr_names, attr_type_list


# create table movie_companies
# (movie_id integer not null, company_id integer not null, company_type_id integer not null);
def _parse_create_sql(create_sql: str):
    terms = create_sql.strip().lower().split("create table")
    info = terms[1].strip()
    lidx = info.find("(")
    table_name = info[0:lidx].strip()
    ridx = info.rfind(")")
    attr_infos_str = info[lidx + 1 : ridx].strip()

    # use regex to clean attr_infos_str, since it may contains '(' and ')'
    attr_infos_str = re.sub(r"\([^)]*\)", "", attr_infos_str)
    attr_infos = attr_infos_str.split(",")

    attr_descs = []
    attr_extra_infos = []
    for attr_info in attr_infos:
        terms = attr_info.strip().split(" ")
        attr_name = terms[0].strip()
        data_type = terms[1].strip()
        extra_info = None
        if len(terms) > 2:
            extra_info = " ".join(terms[2:])

        attr_extra_infos.append(extra_info)

        # assert data_type in {'bigint', 'integer', 'character', 'double', 'smallint', 'timestamp', 'serial'}
        if data_type == "character":
            varying_str = terms[2].strip()
            assert varying_str.startswith("varying")

        # assert (data_type == 'integer')
        attr_descs.append((attr_name, data_type))

    return table_name, attr_descs, attr_extra_infos


def _gen_db_info_json(
    table_names: List[str],
    table_attr_types_map: Dict,
    table_attr_map: Dict,
    index_table_map: Dict,
    data_dir: str,
):
    """
    Summarizes and dumps database table information into a JSON file.

    @param table_names: List of table names in the database
    @param table_attr_types_map: Dictionary mapping table names to their respective attribute types
    @param data_dir: Directory where the original table CSV files are stored.
    @param tables_info_path: Path to the output JSON file.
    @return: None

    Output JSON Structure:
        - table_no_map: A dictionary mapping each table name (in lowercase) to its respective index (table number).
        - table_size_list: A list of integers representing the size (number of rows) of each table.
        - attr_no_map_list: A list of dictionaries, each mapping attribute names to their respective indices for each table.
        - attr_no_types_list: A list of lists, each containing the attribute types for the attributes in each table.
        - attr_ranges_list: A list of lists, each containing the range (minimum and maximum values) for the attributes in each table.
        - db_num_attrs: An integer representing the total number of attributes across all tables.
    """

    table_size_list = []

    # {table_name: table_no} {table_no: ori_table_name}
    table_no_map = {}

    # summarize for all tables.
    attr_no_map_list = []
    attr_no_types_list = []
    attr_ranges_list = []

    for i, table_name in enumerate(table_names):
        table_no_map[table_name.lower()] = i

    # read earh table data and summarize the columns info
    db_num_attrs = 0
    for table_name in table_names:
        table_attributes = table_attr_map[table_name]
        attr_type_list = table_attr_types_map[table_name]
        table_file_path = os.path.join(data_dir, table_name + ".csv")
        attr_no_types, table_attr_ranges, table_size = _get_table_info_csv(
            table_attributes, table_file_path, attr_type_list
        )
        attr_no_map_list.append(
            {table_attributes[i]: int(i) for i in range(len(table_attributes))}
        )
        attr_no_types_list.append(attr_no_types)
        attr_ranges_list.append(table_attr_ranges)
        table_size_list.append(table_size)
        db_num_attrs += len(table_attributes)

    attr_no_types_list = [
        attr_no_types.tolist() for attr_no_types in attr_no_types_list
    ]
    attr_ranges_list = [attr_ranges.tolist() for attr_ranges in attr_ranges_list]

    table_info_json = {
        "table_no_map": table_no_map,
        "table_size_list": table_size_list,
        "attr_no_map_list": attr_no_map_list,
        "table_attr_map": table_attr_map,
        "index_table_map": index_table_map,
        "attr_no_types_list": attr_no_types_list,
        "attr_ranges_list": attr_ranges_list,
        "db_num_attrs": db_num_attrs,
    }

    return table_info_json


def load_db_info_json(tables_info_path) -> DBTableInfo:
    # Read the JSON file and parse it into a dictionary
    with open(tables_info_path, "r") as reader:
        table_info = json.load(reader)

    # Convert lists back to numpy arrays where necessary
    table_no_map = table_info["table_no_map"]
    table_size_list = table_info["table_size_list"]
    attr_no_map_list = table_info["attr_no_map_list"]
    table_attr_map = {}
    if "table_attr_map" in table_info:
        table_attr_map = table_info["table_attr_map"]

    attr_no_types_list = []
    if "attr_no_types_list" in table_info:
        attr_no_types_list = [
            np.array(attr_no_types, dtype=np.int64)
            for attr_no_types in table_info["attr_no_types_list"]
        ]

    attr_ranges_list = []
    if "attr_ranges_list" in table_info:
        attr_ranges_list = [
            np.array(attr_ranges, dtype=np.float64)
            for attr_ranges in table_info["attr_ranges_list"]
        ]

    db_num_attrs = {}
    index_table_map = {}
    if "db_num_attrs" in table_info:
        db_num_attrs = table_info["db_num_attrs"]
    if "index_table_map" in table_info:
        index_table_map = table_info["index_table_map"]
    # get the number, table name mapper
    value_to_keys = defaultdict(list)
    for key, value in table_no_map.items():
        value_to_keys[value].append(key)
    no_table_map = {
        value: keys for value, keys in value_to_keys.items() if len(keys) > 1
    }

    result = DBTableInfo(
        table_no_map,
        no_table_map,
        table_size_list,
        attr_no_map_list,
        attr_no_types_list,
        attr_ranges_list,
        index_table_map,
        table_attr_map,
        db_num_attrs,
    )

    return result


def _get_table_info_csv(
    attr_names: list, table_path: str, attr_type_list
) -> (dict, np.ndarray, np.ndarray, int):
    """
    Summarize the attribute id, types, range, and size of the table.
    return: attr_no_map, attr_types, attr_ranges, table_size
    """
    # attr_type_list: the list of the attribute types, 0 for int, 1 for float, -1 for string
    assert attr_type_list is not None
    assert len(attr_names) == len(
        attr_type_list
    ), "The number of attribute names must match the attribute types."

    df = pd.read_csv(
        table_path,
        quotechar='"',
        quoting=csv.QUOTE_ALL,
        on_bad_lines="skip",
        header=None,
        names=attr_names,
        low_memory=False,
    )

    if len(attr_names) != len(df.columns):
        raise ValueError(
            f"{table_path}: Number of columns in CSV does not match the number of attribute names provided."
        )

    card = len(df)

    epsilons = []
    for attr_type in attr_type_list:
        if attr_type == 0:
            epsilons.append(0.5)
        elif attr_type == 1:
            epsilons.append(1e-9)
        else:
            epsilons.append(0)

    attr_no_types = np.array(attr_type_list, dtype=np.int64)
    table_attr_ranges = []

    for i, col in enumerate(df.columns):
        if attr_type_list[i] != -1:
            numeric_data = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()

            if len(numeric_data) > 0:
                minv, maxv = (
                    np.min(numeric_data) - epsilons[i],
                    np.max(numeric_data) + epsilons[i],
                )
            else:
                minv, maxv = 0, 0
            table_attr_ranges.append([minv, maxv])
        else:
            table_attr_ranges.append([0, 0])

    table_attr_ranges = np.array(table_attr_ranges, dtype=np.float64)

    return attr_no_types, table_attr_ranges, card
