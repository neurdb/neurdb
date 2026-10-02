import io
import os
from unittest.mock import patch

from experience.store import content_hash
from optimization.actions import ActionProfile, global_experience_path, stable_state
from optimization.naming import ResultDictReader, environ, legacy_profile
from runtime.policies.fixed import predict


def test_environment_aliases_prefer_new_names_without_mutation():
    with patch.dict(os.environ, {"NQO_FIXED_ENUM": "top5"}, clear=True):
        assert environ["NEURQO_FIXED_ENUM"] == "top5"
        assert "NEURQO_FIXED_ENUM" in environ
        assert predict({"request_type": "enum"})["enum_action"] == "top5"
        assert "NEURQO_FIXED_ENUM" not in os.environ
        os.environ["NEURQO_FIXED_ENUM"] = "native"
        assert predict({"request_type": "enum"})["enum_action"] == "native"


def test_profile_loading_and_cache_hashes_preserve_legacy_identity():
    profile = ActionProfile.from_mapping(name="pg", values={"nqo_enabled": False})
    assert profile.is_postgres
    assert "neurqo_enabled" in profile.to_dict()
    assert "nqo_enabled" not in profile.to_dict()
    assert "nqo_enabled" in profile.legacy_dict()
    assert (
        content_hash(legacy_profile(profile.to_dict())) in profile.compatible_hashes()
    )
    assert content_hash(profile.legacy_dict()) in profile.compatible_hashes()
    assert all(key.startswith("neurqo.") for key in profile.guc_settings())


def test_state_hash_format_does_not_change_with_public_name():
    # This persisted token is deliberately not a public system label.
    assert stable_state({"sql": "SELECT * FROM temp123"}) == {
        "sql": "SELECT * FROM __nqo_temp_1"
    }


def test_csv_method_alias_does_not_change_sql_or_paths():
    reader = ResultDictReader(
        io.StringIO(
            "method,sql,checkpoint\nNQO,SELECT nqo FROM t,results/nqo/best.pt\n"
            "NeurQO,SELECT 1,results/models/best.pt\n"
        )
    )
    first, second = list(reader)
    assert first["method"] == second["method"] == "NeurQO"
    assert first["sql"] == "SELECT nqo FROM t"
    assert first["checkpoint"] == "results/nqo/best.pt"


def test_csv_legacy_runtime_header():
    reader = ResultDictReader(io.StringIO("nqo_total_ms\n42\n"))
    assert reader.fieldnames == ["neurqo_total_ms"]
    assert next(reader)["neurqo_total_ms"] == "42"


def test_existing_buffer_is_reused_without_renaming(tmp_path):
    old = tmp_path / ".nqo_runtime" / "experience" / "job_light.sql"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"immutable")
    assert global_experience_path(tmp_path, "job") == old
    assert old.read_bytes() == b"immutable"
