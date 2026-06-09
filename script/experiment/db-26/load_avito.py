#!/usr/bin/env python3
"""Export the cleaned rel-avito parquet tables to CSV and emit NeurDB DDL + \\copy SQL.

Form B (prebuilt cleaned DB) is used: dates are real timestamps, Params/SearchParams
already dropped, rows filtered to >= 2015-04-25.

Run with an env that has duckdb (e.g. the `neurbench` conda env):
    python load_avito.py
Then load into NeurDB from inside the container:
    docker exec neurdb_dev /code/neurdb-dev/build/psql/bin/psql -h 0.0.0.0 -U neurdb -d neurdb \
        -v ON_ERROR_STOP=1 -f /code/neurdb-dev/test/avito/rel-avito-db/avito_schema.sql \
        -f /code/neurdb-dev/test/avito/rel-avito-db/avito_copy.sql
"""
import os

import duckdb

HOST_DB = "/home/worker/r/neurdb/neurdb-dev/test/avito/rel-avito-db/db"
HOST_OUT = "/home/worker/r/neurdb/neurdb-dev/test/avito/rel-avito-db"
HOST_CSV = os.path.join(HOST_OUT, "csv")
# Same files seen from inside the container (bind mount):
CONTAINER_CSV = "/code/neurdb-dev/test/avito/rel-avito-db/csv"

SCHEMA = "avito"

# (table, [(db_col, out_col, pg_type)], pkey, copy_order)
TABLES = {
    "Category": (
        [
            ("CategoryID", "categoryid", "BIGINT"),
            ("Level", "level", "BIGINT"),
            ("ParentCategoryID", "parentcategoryid", "BIGINT"),
            ("SubcategoryID", "subcategoryid", "BIGINT"),
        ],
        "categoryid",
    ),
    "Location": (
        [
            ("LocationID", "locationid", "BIGINT"),
            ("Level", "level", "DOUBLE PRECISION"),
            ("RegionID", "regionid", "DOUBLE PRECISION"),
            ("CityID", "cityid", "DOUBLE PRECISION"),
        ],
        "locationid",
    ),
    "UserInfo": (
        [
            ("UserID", "userid", "BIGINT"),
            ("UserAgentID", "useragentid", "DOUBLE PRECISION"),
            ("UserAgentOSID", "useragentosid", "DOUBLE PRECISION"),
            ("UserDeviceID", "userdeviceid", "DOUBLE PRECISION"),
            ("UserAgentFamilyID", "useragentfamilyid", "DOUBLE PRECISION"),
        ],
        "userid",
    ),
    "AdsInfo": (
        [
            ("AdID", "adid", "BIGINT"),
            ("LocationID", "locationid", "BIGINT"),
            ("CategoryID", "categoryid", "BIGINT"),
            ("Price", "price", "DOUBLE PRECISION"),
            ("Title", "title", "TEXT"),
            ("IsContext", "iscontext", "DOUBLE PRECISION"),
        ],
        "adid",
    ),
    "SearchInfo": (
        [
            ("SearchID", "searchid", "BIGINT"),
            ("UserID", "userid", "BIGINT"),
            ("SearchDate", "searchdate", "TIMESTAMP"),
            ("IPID", "ipid", "DOUBLE PRECISION"),
            ("IsUserLoggedOn", "isuserloggedon", "DOUBLE PRECISION"),
            ("SearchQuery", "searchquery", "TEXT"),
            ("LocationID", "locationid", "BIGINT"),
            ("CategoryID", "categoryid", "BIGINT"),
        ],
        "searchid",
    ),
    "SearchStream": (
        [
            ("SearchID", "searchid", "BIGINT"),
            ("AdID", "adid", "BIGINT"),
            ("Position", "position", "DOUBLE PRECISION"),
            ("ObjectType", "objecttype", "DOUBLE PRECISION"),
            ("HistCTR", "histctr", "DOUBLE PRECISION"),
            ("IsClick", "isclick", "DOUBLE PRECISION"),
            ("SearchDate", "searchdate", "TIMESTAMP"),
        ],
        None,
    ),
    "VisitStream": (
        [
            ("UserID", "userid", "BIGINT"),
            ("IPID", "ipid", "DOUBLE PRECISION"),
            ("AdID", "adid", "BIGINT"),
            ("ViewDate", "viewdate", "TIMESTAMP"),
        ],
        None,
    ),
    "PhoneRequestsStream": (
        [
            ("UserID", "userid", "BIGINT"),
            ("IPID", "ipid", "DOUBLE PRECISION"),
            ("AdID", "adid", "BIGINT"),
            ("PhoneRequestDate", "phonerequestdate", "TIMESTAMP"),
        ],
        None,
    ),
}

INDEXES = [
    ("searchstream", "adid"),
    ("searchstream", "searchid"),
    ("searchstream", "searchdate"),
    ("searchinfo", "searchdate"),
    ("searchinfo", "userid"),
    ("visitstream", "adid"),
    ("visitstream", "viewdate"),
    ("phonerequestsstream", "adid"),
    ("phonerequestsstream", "phonerequestdate"),
    ("adsinfo", "categoryid"),
    ("adsinfo", "locationid"),
]


def main():
    os.makedirs(HOST_CSV, exist_ok=True)
    con = duckdb.connect()
    ddl, copy = [f"CREATE SCHEMA IF NOT EXISTS {SCHEMA};", ""], []
    for tbl, (cols, pk) in TABLES.items():
        lt = tbl.lower()
        src = f"{HOST_DB}/{tbl}.parquet"
        csv = f"{HOST_CSV}/{lt}.csv"
        sel = ", ".join(f'"{c[0]}" AS {c[1]}' for c in cols)
        con.execute(
            f"COPY (SELECT {sel} FROM '{src}') TO '{csv}' "
            "(FORMAT CSV, HEADER, DELIMITER ',')"
        )
        n = con.execute(f"SELECT count(*) FROM '{src}'").fetchone()[0]
        print(f"exported {tbl:20s} -> {lt}.csv  rows={n:,}")
        coldefs = ",\n  ".join(f"{c[1]} {c[2]}" for c in cols)
        pkline = f",\n  PRIMARY KEY ({pk})" if pk else ""
        ddl.append(f"DROP TABLE IF EXISTS {SCHEMA}.{lt} CASCADE;")
        ddl.append(f"CREATE TABLE {SCHEMA}.{lt} (\n  {coldefs}{pkline}\n);")
        collist = ", ".join(c[1] for c in cols)
        copy.append(
            f"\\copy {SCHEMA}.{lt} ({collist}) FROM "
            f"'{CONTAINER_CSV}/{lt}.csv' WITH (FORMAT csv, HEADER true);"
        )
    for t, c in INDEXES:
        ddl.append(f"-- index after load")
        copy.append(f"CREATE INDEX IF NOT EXISTS idx_{t}_{c} ON {SCHEMA}.{t} ({c});")

    with open(f"{HOST_OUT}/avito_schema.sql", "w") as f:
        f.write("\n".join(ddl) + "\n")
    with open(f"{HOST_OUT}/avito_copy.sql", "w") as f:
        f.write("\n".join(copy) + "\n")
    print(f"\nwrote {HOST_OUT}/avito_schema.sql and avito_copy.sql")


if __name__ == "__main__":
    main()
