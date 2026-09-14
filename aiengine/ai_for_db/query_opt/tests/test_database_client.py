from unittest.mock import MagicMock

import psycopg2
import pytest
from database.client import configure_session, dataset_settings, execute_sql, main


def test_dataset_parameters_are_packaged():
    assert dataset_settings("job")["nqo.aja_conservative_rows"] == 100000
    assert dataset_settings("stack")["nqo.lip_max_build_relation_rows"] == 10000
    assert dataset_settings("tpch")["nqo.aja_aggressive_rows"] == 10000


def test_enable_only_after_session_configuration():
    cursor = MagicMock()
    configure_session(
        cursor,
        enabled=True,
        server_url="http://127.0.0.1:8088/action",
        timeout_ms=60000,
        dataset="job",
    )
    calls = cursor.execute.call_args_list
    assert calls[0].args[1] == ("off",)
    assert "nqo" in repr(calls[0].args[0])
    assert calls[1].args[1] == (60000,)
    assert calls[-1].args == ("SET nqo TO on",)
    assert any("nqo.server_url" in repr(call.args[0]) for call in calls)


def test_pg_does_not_set_action_parameters():
    cursor = MagicMock()
    configure_session(
        cursor, enabled=False, server_url="unused", timeout_ms=1000, dataset="job"
    )
    assert cursor.execute.call_count == 2


def test_execute_once_includes_result_fetch(monkeypatch):
    clock = iter([10.0, 10.125])
    monkeypatch.setattr("database.client.time.perf_counter", lambda: next(clock))
    cursor = MagicMock(description=[("v",)], rowcount=1)
    cursor.fetchall.return_value = [(1,)]
    result = execute_sql(cursor, "SELECT 1")
    cursor.execute.assert_called_once_with("SELECT 1")
    cursor.fetchall.assert_called_once()
    assert result == {
        "wall_ms": 125.0,
        "columns": ["v"],
        "rows": [(1,)],
        "row_count": 1,
    }


def test_execute_ddl_has_no_result_fetch():
    cursor = MagicMock(description=None, rowcount=-1)
    result = execute_sql(cursor, "CREATE TEMP TABLE t(v int)")
    assert result["rows"] == []
    cursor.fetchall.assert_not_called()


def test_timeout_returns_error_and_closes_session(monkeypatch, capsys):
    connection = MagicMock()
    cursor = connection.cursor.return_value.__enter__.return_value

    def execute(statement, *args):
        if statement == "SELECT pg_sleep(2)":
            raise psycopg2.errors.QueryCanceled("statement timeout")

    cursor.execute.side_effect = execute
    monkeypatch.setattr("database.client.psycopg2.connect", lambda **kw: connection)
    assert main(["--sql", "SELECT pg_sleep(2)", "--timeout-ms", "1"]) == 1
    assert "statement timeout" in capsys.readouterr().err
    connection.close.assert_called_once()


def test_catalog_export_does_not_enable_nqo(monkeypatch, tmp_path):
    connection = MagicMock()
    cursor = connection.cursor.return_value.__enter__.return_value
    monkeypatch.setattr("database.client.psycopg2.connect", lambda **kw: connection)
    monkeypatch.setattr("database.client.read_postgres_catalog", lambda conn: {})
    write = MagicMock()
    monkeypatch.setattr("database.client.write_catalog_snapshot", write)
    path = tmp_path / "catalog.json"
    assert main(["--export-catalog", str(path)]) == 0
    write.assert_called_once_with(path, {})
    assert not any(
        call.args == ("SET nqo TO on",) for call in cursor.execute.call_args_list
    )
    connection.close.assert_called_once()


@pytest.mark.parametrize("arguments", [[], ["--sql", "SELECT 1", "--timeout-ms", "0"]])
def test_missing_input_or_invalid_timeout(arguments):
    with pytest.raises(SystemExit, match="2"):
        main(arguments)
