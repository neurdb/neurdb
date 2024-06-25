import re


def parse_conditions(input_str):
    # Extract the field names
    fields = re.findall(r'\"([^\"]+)\"\s*\"([^\"]+)\"\)', input_str)
    # Extract the values
    values = re.findall(r':val\s*\"([^\"]+)\"', input_str)

    # Combine fields and values into a condition string
    conditions = []
    for (table, column), value in zip(fields, values):
        conditions.append(f"{table}.{column} = '{value}'")

    # Join all conditions with 'AND'
    return " AND ".join(conditions)

# input_str = "{BOOLEXPR :boolop and :args ({A_EXPR :name (\"=\") :lexpr {COLUMNREF :fields (\"cn\" \"country_code\") :location 74} :rexpr {A_CONST :val \"[us]\" :location 91} :location 90} {A_EXPR :name (\"=\") :lexpr {COLUMNREF :fields (\"k\" \"keyword\") :location 104} :rexpr {A_CONST :val \"character-name-in-title\" :location 115} :location 114} {A_EXPR :name (\"=\") :lexpr {COLUMNREF :fields (\"cn\" \"id\") :location 147} :rexpr {COLUMNREF :fields (\"mc\" \"company_id\") :location 155} :location 153}) :location 100}"
# print(parse_conditions(input_str))
