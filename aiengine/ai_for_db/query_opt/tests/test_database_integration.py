"""Opt-in, bounded smoke tests against the development JOB database."""

import json
import os
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from urllib.error import URLError
from urllib.request import urlopen

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("NQO_TEST_DB") != "1", reason="set NQO_TEST_DB=1 for DB smoke tests"
)

QUERY = """
SELECT MIN(t.title) AS movie_title
FROM company_name AS cn, keyword AS k, movie_companies AS mc,
     movie_keyword AS mk, title AS t
WHERE cn.country_code = '[de]'
  AND k.keyword = 'character-name-in-title'
  AND cn.id = mc.company_id AND mc.movie_id = t.id
  AND t.id = mk.movie_id AND mk.keyword_id = k.id
  AND mc.movie_id = mk.movie_id
"""


@contextmanager
def action_server(directory, *, fixed=None, model=None, catalog=None):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    url = f"http://127.0.0.1:{port}/action"
    env = {
        key: value for key, value in os.environ.items() if not key.startswith("NQO_")
    }
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    command = [
        sys.executable,
        "-m",
        "runtime.action_server",
        "--port",
        str(port),
        "--require-model",
        "--trajectory-log",
        str(directory / "policy.jsonl"),
    ]
    if model:
        command += [
            "--model-path",
            model,
            "--catalog-path",
            catalog,
            "--workload",
            "job",
        ]
    else:
        command += ["--model-module", "runtime.policies.fixed:predict"]
        env.update(fixed or {})
    with (directory / "server.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log, env=env, cwd="/tmp")
        try:
            deadline = time.monotonic() + 20
            while True:
                if process.poll() is not None or time.monotonic() > deadline:
                    raise AssertionError((directory / "server.log").read_text())
                try:
                    with urlopen(url, timeout=1) as response:
                        health = response.read().decode()
                    assert "model_source=stub" not in health
                    break
                except URLError:
                    time.sleep(0.05)
            yield url
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def run_query(url, log, *, pg=False):
    # Exercise the installed client entry point, not an in-process substitute.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "database.client",
            "--dataset",
            "job",
            "--server-url",
            url,
            "--trajectory-log",
            str(log),
            "--sql",
            QUERY,
        ]
        + (["--pg"] if pg else []),
        capture_output=True,
        text=True,
        timeout=70,
        cwd="/tmp",
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.fixture(scope="module")
def pg_result(tmp_path_factory):
    return run_query(
        "http://127.0.0.1:1/action", tmp_path_factory.mktemp("pg") / "none", pg=True
    )


@pytest.mark.parametrize("mode", ["native", "split", "top5", "lip", "aja", "combined"])
def test_fixed_policy_through_postgres(mode, tmp_path, pg_result):
    fixed = {
        "NQO_FIXED_DEC": "apply" if mode == "split" else "skip",
        "NQO_FIXED_ENUM": "top5" if mode in ("top5", "combined") else "native",
        "NQO_FIXED_FILTER": "selective" if mode in ("lip", "combined") else "none",
        "NQO_FIXED_AJOIN": "conservative" if mode in ("aja", "combined") else "off",
        "NQO_FIXED_SCHED_ALPHA": "0.5",
    }
    with action_server(tmp_path, fixed=fixed) as url:
        result = run_query(url, tmp_path / "db.jsonl")
    assert result["rows"] == pg_result["rows"]
    events = [
        json.loads(line) for line in (tmp_path / "db.jsonl").read_text().splitlines()
    ]
    assert events[-1]["phase"] == "final"
    decisions = [
        json.loads(line)
        for line in (tmp_path / "policy.jsonl").read_text().splitlines()
    ]
    phases = [entry["state"]["request_type"] for entry in decisions]
    assert {"dec", "enum", "adapt"}.issubset(phases)
    for entry in decisions:
        state = entry["state"]
        if state["request_type"] in ("dec", "enum"):
            assert "plan_json" not in state
        if state["request_type"] == "adapt":
            assert "plan_json" in state
            assert "sql" not in state and "relations" not in state
    if mode == "split":
        assert "sched" in phases
        assert any(event["phase"] == "split" for event in events)
    if mode in ("top5", "combined"):
        assert any(event["action"]["search_applied"] for event in events)
    if mode == "lip":
        assert any(event["action"]["lip_filters"] > 0 for event in events)
    if mode == "aja":
        assert any(event["aja"]["decided"] > 0 for event in events)


def test_trusted_checkpoint_through_postgres(tmp_path, pg_result):
    model = os.environ.get("NQO_TEST_MODEL")
    catalog = os.environ.get("NQO_TEST_CATALOG")
    if not model or not catalog:
        pytest.skip("provide NQO_TEST_MODEL and NQO_TEST_CATALOG for checkpoint smoke")
    with action_server(tmp_path, model=model, catalog=catalog) as url:
        result = run_query(url, tmp_path / "db.jsonl")
    assert result["rows"] == pg_result["rows"]
    decisions = [
        json.loads(line)
        for line in (tmp_path / "policy.jsonl").read_text().splitlines()
    ]
    assert decisions
    assert all(
        entry["action"]["model_source"].startswith("checkpoint:") for entry in decisions
    )
