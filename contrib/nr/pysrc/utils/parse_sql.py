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
