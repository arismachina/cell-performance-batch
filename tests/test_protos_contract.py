"""Guards on the Protos-facing contract: protos.toml + the wrapper.

These are the parts nothing else exercises until an import fails in the
platform, where the feedback loop is a container build long.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PROTOS_TOML = REPO_ROOT / "protos.toml"
WRAPPER = REPO_ROOT / "protos" / "wrapper.py"


@pytest.fixture(scope="module")
def spec() -> dict:
    return tomllib.loads(PROTOS_TOML.read_text())


@pytest.fixture(scope="module")
def wrapper_module():
    spec = importlib.util.spec_from_file_location("protos_wrapper", WRAPPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestProtosToml:
    def test_declares_the_supported_schema_version(self, spec):
        assert spec["schema_version"] == 1

    def test_single_model_python_declaration(self, spec):
        assert isinstance(spec["model"], dict), "single [model], not [[model]]"
        assert spec["model"]["language"] == "python"
        assert spec["model"]["name"]
        assert spec["model"]["description"].strip()

    def test_wrapper_path_exists_in_the_repo(self, spec):
        declared = REPO_ROOT / spec["wrapper"]["path"]
        assert declared.is_file(), f"{declared} is declared but missing"
        assert declared.resolve() == WRAPPER.resolve()

    def test_schemas_are_inline(self, spec):
        # Protos parses input_schema_path / output_schema_path but never
        # resolves them, so a path-based declaration silently ships an
        # empty schema and the Execute Model UI falls back to a raw JSON
        # textarea. Inline is the only form that reaches the platform.
        wrapper = spec["wrapper"]
        assert "input_schema_path" not in wrapper
        assert "output_schema_path" not in wrapper
        assert wrapper["input_schema"]["properties"]["designs"]["type"] == "array"
        assert wrapper["output_schema"]["properties"]["results"]["type"] == "array"

    def test_designs_is_the_only_required_input(self, spec):
        assert spec["wrapper"]["input_schema"]["required"] == ["designs"]

    def test_result_detail_enum_matches_the_batch_model(self, spec):
        from cell_performance_batch.batch import BatchInput

        declared = set(
            spec["wrapper"]["input_schema"]["properties"]["result_detail"]["enum"]
        )
        implemented = set(
            BatchInput.model_json_schema()["properties"]["result_detail"]["enum"]
        )
        assert declared == implemented

    def test_max_workers_ceiling_matches_the_batch_cap(self, spec):
        from cell_performance_batch.batch import MAX_WORKERS_CAP

        declared = spec["wrapper"]["input_schema"]["properties"]["max_workers"]
        assert declared["maximum"] == MAX_WORKERS_CAP


class TestWrapperContract:
    def _run(self, model_input: str | None) -> subprocess.CompletedProcess:
        env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
        if model_input is None:
            env.pop("MODEL_INPUT", None)
        else:
            env["MODEL_INPUT"] = model_input
        return subprocess.run(
            [sys.executable, str(WRAPPER)],
            capture_output=True,
            text=True,
            env=env,
            timeout=300,
        )

    def test_empty_batch_exits_clean_with_one_json_line(self):
        # Mirrors the platform's dry-run validation: sample input
        # generated from the schema is {"designs": [], ...}.
        proc = self._run(json.dumps({"designs": [], "simulation_parameters": {}}))
        assert proc.returncode == 0, proc.stderr
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
        assert payload["results"] == []
        assert payload["summary"]["design_count"] == 0

    def test_missing_model_input_fails_loudly(self):
        proc = self._run(None)
        assert proc.returncode == 1
        assert "MODEL_INPUT" in proc.stderr

    def test_malformed_model_input_fails_loudly(self):
        proc = self._run("not json")
        assert proc.returncode == 1
        assert "not valid JSON" in proc.stderr

    def test_invalid_envelope_still_emits_json_and_exits_nonzero(self):
        proc = self._run(json.dumps({"designs": [{"name": "no cell_parameters"}]}))
        assert proc.returncode == 1
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
        assert "error" in payload

    @pytest.mark.slow()
    def test_real_run_keeps_stdout_to_a_single_json_line(self):
        # The vendored model logs its solve steps to stdout; the wrapper
        # has to keep those off the result stream.
        example = json.loads((REPO_ROOT / "examples" / "two_designs.json").read_text())
        proc = self._run(json.dumps(example))
        assert proc.returncode == 0, proc.stderr
        assert len(proc.stdout.strip().splitlines()) == 1
        payload = json.loads(proc.stdout)
        assert payload["summary"]["succeeded"] == 2
        assert "Evaluating design" in proc.stderr


class TestJsonSafety:
    def test_non_finite_floats_become_null(self, wrapper_module):
        cleaned = wrapper_module._json_safe(
            {
                "kpis": [{"a": float("nan"), "b": float("inf"), "c": 1.5}],
                "name": "keep me",
            }
        )
        assert cleaned["kpis"][0] == {"a": None, "b": None, "c": 1.5}
        assert cleaned["name"] == "keep me"
        # And the result is now serialisable under strict JSON.
        json.dumps(cleaned, allow_nan=False)
