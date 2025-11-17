# Standard library imports
from collections import namedtuple

# Define the named tuple with comments for each field
DBTableInfo = namedtuple(
    "DBTableInfo",
    [
        "table_no_map",  # Mapping of table names (lowercase) to table numbers
        "no_table_map",  # Reverse mapping of table numbers to original table names
        "table_size_list",  # List of table sizes (number of rows) for each table
        "attr_no_map_list",  # List of attribute-to-index mappings for each table
        "attr_no_types_list",  # List of attribute types for each table (as numpy arrays)
        # List of attribute value ranges for each table (as numpy arrays),
        # each element is an n x 2 numpy float matrix.
        "attr_ranges_list",
        "index_table_map",
        "table_attr_map",
        "db_num_attrs",  # Total number of attributes across all tables
    ],
)


WorkloadQueryInfo = namedtuple(
    "WorkloadQueryInfo",
    [
        # List of possible join attributes N * 4 numpy int matrx with each row looks
        # like [table_no1, table_no1.attr_no, table_no2, table_no2.attr_no]
        "possible_join_attrs",
        "join_conds_list",  # List of join conditions
        "filter_attr_range_conds_list",  # List of attribute range conditions
        "join_type_list",  # List of join types
        "relevant_tables_list",  # List of relevant tables
    ],
)
