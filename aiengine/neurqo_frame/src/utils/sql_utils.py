def is_write_sql(line):
    return (
        line.startswith("insert")
        or line.startswith("delete")
        or line.startswith("update")
    )


def is_read_sql(line):
    return line.startswith("select")
