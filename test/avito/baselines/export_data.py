#!/usr/bin/env python3
"""Export the avito tables needed by the AdCTR workload from NeurDB to parquet.

This is the EXPORT step of the export-execute-import baseline pattern: the
relational data leaves the database once, and the downstream framework
(LOTUS / Palimpzest / plain pandas) works on local files.

Only the columns the workload actually touches are exported (what a competent
engineer would select); see test/avito/workloads/02_features_adctr_pit.sql.

Usage:
    python export_data.py [--out DIR]

Emits one "TIMING,export_<table>,<seconds>" line per table plus a total, so the
driver can fold export cost into the end-to-end comparison.
"""

from __future__ import annotations

import argparse
import io
import os
import time

import pandas as pd
import psycopg2

TABLES = {
    "adsinfo": "SELECT adid, locationid, categoryid, price, title, iscontext FROM adsinfo",
    "searchstream": "SELECT searchid, adid, position, histctr, isclick, searchdate FROM searchstream",
    "visitstream": "SELECT adid, viewdate FROM visitstream",
    "phonerequestsstream": "SELECT adid, phonerequestdate FROM phonerequestsstream",
    "category": "SELECT categoryid, level, parentcategoryid, subcategoryid FROM category",
    "location": "SELECT locationid, level, regionid, cityid FROM location",
}

PARSE_DATES = {
    "searchstream": ["searchdate"],
    "visitstream": ["viewdate"],
    "phonerequestsstream": ["phonerequestdate"],
}


def connect():
    return psycopg2.connect(
        host=os.environ.get("PGHOST", "127.0.0.1"),
        port=int(os.environ.get("PGPORT", "5432")),
        user=os.environ.get("PGUSER", "neurdb"),
        password=os.environ.get("PGPASSWORD", "neurdb"),
        dbname=os.environ.get("PGDATABASE", "avito"),
    )


def export_table(conn, name: str, query: str, out_dir: str) -> float:
    t0 = time.time()
    buf = io.StringIO()
    with conn.cursor() as cur:
        cur.copy_expert(f"COPY ({query}) TO STDOUT WITH CSV HEADER", buf)
    buf.seek(0)
    df = pd.read_csv(buf, parse_dates=PARSE_DATES.get(name))
    df.to_parquet(os.path.join(out_dir, f"{name}.parquet"), index=False)
    dt = time.time() - t0
    print(f"TIMING,export_{name},{dt:.1f}  ({len(df)} rows)", flush=True)
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "data"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    total = 0.0
    with connect() as conn:
        for name, query in TABLES.items():
            total += export_table(conn, name, query, args.out)
    print(f"TIMING,export_total,{total:.1f}", flush=True)


if __name__ == "__main__":
    main()
