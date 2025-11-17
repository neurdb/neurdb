# Standard library imports
import configparser
import json
import os
import time
from math import log
from typing import Any, Dict, List, Optional

# Third-party imports
import psycopg2


class PostgresConnector:
    """Handles connection and operations with PostgreSQL database."""

    class TimedResult:
        """Result wrapper with timing information."""

        def __init__(self, result: str, time: int):
            self.result = result
            self.time_usecs = time

    def __init__(self, database_name: str):
        # Get connection config from config-file
        self.config = configparser.ConfigParser()
        config_path = os.path.dirname(__file__) + "/../../configs/postgres.cfg"
        print(f"Reading config path from {config_path}")
        self.config.read(config_path)
        defaults = self.config["DEFAULT"]
        user = defaults["DB_USER"]
        database = database_name
        password = defaults["DB_PASSWORD"]
        host = defaults["DB_HOST"]
        port = defaults["DB_PORT"]
        self.timeout = defaults["TIMEOUT_MS"]
        self.postgres_connection_string = (
            f"postgresql://{user}:{password}@{host}:{port}/{database}"
        )
        self.connection = None
        self.cursor = None
        self.database_name = database_name
        # Cache for selectivity calculations
        self.latency_record_dict = {}

    def _default_setting(self):
        """Apply default database settings."""
        self.cursor.execute(f"SET statement_timeout TO {self.timeout};")
        self.cursor.execute("ALTER SYSTEM SET autovacuum TO off;")
        self.cursor.execute("SELECT pg_reload_conf();")

    def connect(self) -> None:
        """Establish database connection."""
        try:
            self.connection = psycopg2.connect(self.postgres_connection_string)
            self.connection.autocommit = True
            self.cursor = self.connection.cursor()
            self._default_setting()
        except psycopg2.Error as e:
            print(f"Error connecting to the database: {e}")
            if self.connection:
                self.connection.rollback()
            raise e

    def close(self) -> None:
        """Close database connection."""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def get_server_config(self):
        """Returns all live configs as [(param, value, help)]."""
        self.cursor.execute("show all;")
        return self.cursor.fetchall()

    # -------------------- GEQO Management --------------------

    def set_geqo(self, flag: str) -> None:
        """
        Set GEQO (Genetic Query Optimizer) state.

        Args:
            flag: 'on', 'off', or 'default'
        """
        assert flag in ["on", "off", "default"], f"Invalid flag: {flag}"
        self.cursor.execute(f"SET geqo = {flag};")
        if flag == "on":
            self.cursor.execute("SET geqo_threshold = 12;")

    def set_geqo_exp(self, flag: str) -> None:
        """
        Set GEQO (alias for set_geqo, kept for backward compatibility).

        Args:
            flag: 'on', 'off', or 'default'
        """
        self.set_geqo(flag)

    # -------------------- Parallelism Controls --------------------
    def set_parallel(
        self,
        workers: Optional[int] = None,
        workers_per_gather: Optional[int] = None,
        maintenance_workers: Optional[int] = None,
    ) -> None:
        """
        Adjust session parallelism settings via SET statements.
        Pass integers you want to change; leave None to keep unchanged.
        """
        try:
            stmts = []
            if workers is not None:
                stmts.append(f"SET max_parallel_workers = {int(workers)};")
            if workers_per_gather is not None:
                stmts.append(
                    f"SET max_parallel_workers_per_gather = {int(workers_per_gather)};"
                )
            if maintenance_workers is not None:
                stmts.append(
                    f"SET max_parallel_maintenance_workers = {int(maintenance_workers)};"
                )
            if stmts:
                self.cursor.execute("".join(stmts))
        except psycopg2.Error as e:
            print(f"Error setting parallelism: {e}")
            self.connection.rollback()
            raise e

    def disable_parallel(self) -> None:
        """Convenience: disable all parallel workers for this session."""
        self.set_parallel(workers=0, workers_per_gather=0, maintenance_workers=0)

    # -------------------- Explain and Analysis --------------------
    def apply_hints(self, hints: Optional[List[str]] = None) -> None:
        if not hints:
            return
        for hint in hints:
            self.cursor.execute(hint)

    def explain(self, query: str, geqo: str = "on"):
        """
        Explain (estimate only) a SQL query and return the plan JSON.

        Args:
            query: SQL statement to EXPLAIN
            geqo: GEQO setting ('on', 'off', or 'default')

        Returns:
            Dict containing the plan JSON (estimates only)
        """
        try:
            self.disable_parallel()
            # Set GEQO
            self.set_geqo(geqo)

            # Build EXPLAIN query
            options = ["VERBOSE", "COSTS", "FORMAT JSON"]
            explain_query = f"EXPLAIN ({', '.join(options)}) {query}"

            start = time.time()
            self.cursor.execute(explain_query)
            row = self.cursor.fetchone()
            plan_json = row[0][0] if isinstance(row[0], list) else row[0]

            # Add client-side measured planning time in milliseconds
            plan_time = (time.time() - start) * 1000.0

            return plan_json["Plan"], plan_time
        except psycopg2.Error as e:
            print(f"Error explaining query: {e}")
            self.connection.rollback()
            raise e

    def explain_analysis(self, query: str, geqo_off: bool = False):
        """
        Explain ANALYZE (execute) a SQL query, returning plan JSON and actual latency.

        Args:
            query: SQL statement to EXPLAIN
            geqo_off: If True, disable GEQO

        Returns:
            Tuple of (plan_json, latency_ms)
        """
        try:
            self.disable_parallel()
            geqo = "off" if geqo_off else "on"
            # Set GEQO
            self.set_geqo(geqo)
            # Build EXPLAIN ANALYZE query
            options = [
                "ANALYZE",
                "BUFFERS",
                "VERBOSE",
                "COSTS",
                "FORMAT JSON",
                "SUMMARY",
            ]
            explain_query = f"EXPLAIN ({', '.join(options)}) {query}"
            self.cursor.execute(explain_query)
            row = self.cursor.fetchone()
            plan_json = row[0][0] if isinstance(row[0], list) else row[0]
            # Add client-side measured elapsed time as Planning Time if available is not desired
            # Extract latency from plan
            exec_time = plan_json.get("Execution Time")
            plan_time = plan_json.get("Planning Time")
            return plan_json["Plan"], exec_time, plan_time

        except psycopg2.Error as e:
            print(f"Error explaining + analyzing query: {e}")
            self.connection.rollback()
            raise e

    # -------------------- Database Schema and Statistics --------------------

    def get_selectivity(self, table: str, whereCondition: str) -> float:
        """
        Get selectivity for a given table and where condition.
        Uses caching to avoid repeated calculations.

        Args:
            table: Table name
            whereCondition: WHERE condition string

        Returns:
            float: Negative log of selectivity (-log(select_rows / total_rows))
        """
        # Check cache first
        if whereCondition in self.latency_record_dict:
            return self.latency_record_dict[whereCondition]

        try:
            self.cursor.execute("SET statement_timeout = " + str(int(100000)) + ";")

            # Get total rows
            totalQuery = "select * from " + table + ";"
            self.cursor.execute("EXPLAIN " + totalQuery)
            rows = self.cursor.fetchall()[0][0]
            total_rows = int(rows.split("rows=")[-1].split(" ")[0])
            print(f"total_rows for {table}: {total_rows}")

            # Get selected rows
            resQuery = "select * from " + table + " Where " + whereCondition + ";"
            print("Running sql: EXPLAIN " + resQuery)
            self.cursor.execute("EXPLAIN " + resQuery)
            rows = self.cursor.fetchall()[0][0]
            select_rows = int(rows.split("rows=")[-1].split(" ")[0])

            selectivity = select_rows / total_rows if total_rows > 0 else 0
            result = -log(selectivity) if selectivity > 0 else float("inf")
            print(
                f"select_rows for {whereCondition}: {select_rows}, "
                f"selectivity: {selectivity}, logresult: {result}"
            )

            # Cache the result
            self.latency_record_dict[whereCondition] = result
            return result
        except psycopg2.Error as e:
            print(f"Error calculating selectivity: {e}")
            self.connection.rollback()
            raise e

    def parse_database_schema(self) -> Optional[Dict[str, Any]]:
        """
        Parse all tables, their columns, and row counts from the current database.

        Returns:
            dict: A dictionary containing:
                - table_no_map: Mapping of table names to IDs
                - attr_no_map_list: List of dictionaries mapping column names to IDs for each table
                - table_size_list: List of row counts for each table
        """
        # Check if cached schema info exists
        cache_file = f"./experiment/{self.database_name}_schema_info.json"
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                loaded_db_info = json.load(f)
            print(f"Loaded existing table info from {cache_file}")
            return loaded_db_info

        table_no_map = {}
        attr_no_map_list = []
        table_size_list = []

        try:
            # Get all table names from the current database
            self.cursor.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_type = 'BASE TABLE';
            """
            )
            tables = self.cursor.fetchall()

            # Assign IDs and process each table
            for table_idx, table_tuple in enumerate(tables):
                table_name = table_tuple[0]
                table_no_map[table_name] = table_idx

                # Get column information for this table
                self.cursor.execute(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                    AND table_name = %s
                    ORDER BY ordinal_position;
                """,
                    (table_name,),
                )
                columns = self.cursor.fetchall()

                # Create column mapping for this table
                column_map = {}
                for col_idx, column_tuple in enumerate(columns):
                    column_name = column_tuple[0]
                    column_map[column_name] = col_idx

                attr_no_map_list.append(column_map)

                # Get row count for this table
                self.cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
                row_count = self.cursor.fetchone()[0]
                table_size_list.append(row_count)
                print(
                    f"Table: {table_name}, ID: {table_idx}, Rows: {row_count}, Columns: {len(column_map)}"
                )

        except psycopg2.Error as e:
            print(f"Database error: {e}")
            if self.connection:
                self.connection.rollback()
            return None

        # Update the db_info dictionary
        updated_db_info = {
            "table_no_map": table_no_map,
            "attr_no_map_list": attr_no_map_list,
            "table_size_list": table_size_list,
        }

        # Write the updated dictionary to a JSON file
        with open(cache_file, "w") as f:
            json.dump(updated_db_info, f, indent=4)

        return updated_db_info

    def drop_buffer_cache(self) -> None:
        """
        Drop buffer cache (DISCARD ALL).
        This will discard everythig, including SET.. etc
        """
        self.cursor.execute("DISCARD ALL;")
