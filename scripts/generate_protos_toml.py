"""Regenerate protos.toml with the full model structs inlined.

    python scripts/generate_protos_toml.py           # write protos.toml
    python scripts/generate_protos_toml.py --check   # fail if it's stale

Protos reads ``[wrapper.input_schema]`` / ``[wrapper.output_schema]``
straight out of ``protos.toml`` — it parses ``input_schema_path`` but
never resolves it, so a path-declared schema ships empty. That means
the full ``CellParametersInput`` / ``SimulationParameters`` /
``CellPerformanceKPIs`` structs have to be *in* the TOML if the
platform is to see them. Hand-maintaining ~130 fields of TOML against a
vendored model that moves upstream is not a thing anyone should do, so
this script derives them from the Pydantic models themselves.

Three transforms happen on the way from Pydantic to TOML:

- **``$ref`` / ``$defs`` are inlined.** Protos' schema consumers (the
  dry-run sample generator, the Execute Model form) walk ``properties``
  directly and don't resolve references, so a ref-carrying schema
  reaches them as an opaque object. Cycles raise rather than recurse.
- **Null values are dropped.** TOML has no null. In practice this only
  hits ``"default": null`` on optional fields, where "no default key"
  and "defaults to null" mean the same thing to a form renderer.
- **Auto-generated ``title`` keys are dropped** — see :func:`_drop_titles`.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
TOML_PATH = REPO_ROOT / "protos.toml"

sys.path.insert(0, str(REPO_ROOT / "src"))

# Everything above the generated schemas. Kept verbatim so identity,
# language and the wrapper path stay hand-owned.
HEADER = '''# Aris Protos wrapper declaration — see docs/WRAPPER_SPEC.md in protos-v2.
# Importing this repo via Models Library -> GitHub picks this file up and
# skips LLM wrapper drafting.
#
# THE SCHEMAS BELOW ARE GENERATED. Edit the Pydantic models, then run
#   python scripts/generate_protos_toml.py
# `pytest tests/test_protos_contract.py` fails if this file goes stale.
schema_version = 1

[model]
name = "Cell Performance Batch"
description = """
Evaluates an array of cell designs and returns an array of performance \\
KPI sets — capacity, energy, energy/power density, DCIR, mass, volume, \\
N/P ratio and OCV limits — one entry per design, in input order. Wraps \\
the same equilibrium + PyBaMM SPMe pipeline as the single-design \\
cell_performance model, with per-design failure isolation so one bad \\
point never kills the sweep.
"""
language = "python"
license = "Proprietary"

[wrapper]
path = "protos/wrapper.py"
'''


# ── schema assembly ──────────────────────────────────────────────────


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Return ``schema`` with every ``$ref`` replaced by its definition."""
    defs = schema.get("$defs", {})

    def resolve(node: Any, seen: tuple[str, ...]) -> Any:
        if isinstance(node, list):
            return [resolve(item, seen) for item in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            ref = node["$ref"]
            name = ref.rsplit("/", 1)[-1]
            if not ref.startswith("#/$defs/") or name not in defs:
                raise ValueError(f"Unresolvable $ref: {ref!r}")
            if name in seen:
                raise ValueError(
                    f"Recursive $ref chain {' -> '.join((*seen, name))} — "
                    "inlining would not terminate."
                )
            target = resolve(defs[name], (*seen, name))
            # Sibling keys (description, default, ...) win over the
            # definition's own, matching JSON Schema 2020-12 semantics.
            siblings = {k: resolve(v, seen) for k, v in node.items() if k != "$ref"}
            return {**target, **siblings}
        return {key: resolve(value, seen) for key, value in node.items()}

    resolved = resolve({k: v for k, v in schema.items() if k != "$defs"}, ())
    return resolved


def _drop_nulls(node: Any) -> Any:
    """Strip null values — TOML can't express them."""
    if isinstance(node, dict):
        return {k: _drop_nulls(v) for k, v in node.items() if v is not None}
    if isinstance(node, list):
        if any(item is None for item in node):
            raise ValueError(f"null inside a list has no TOML form: {node!r}")
        return [_drop_nulls(item) for item in node]
    return node


def _drop_titles(node: Any) -> Any:
    """Strip Pydantic's auto-generated ``title`` keys.

    Pydantic titleises the field name — ``positive_electrode_mass_loading_mg_cm2``
    becomes "Positive Electrode Mass Loading Mg Cm2", which mangles the
    units a form renderer would show. The hand-written ``description``
    ("Positive mass loading [mg/cm²]") is strictly better, and dropping
    the titles takes a meaningful bite out of the file.
    """
    if isinstance(node, dict):
        return {k: _drop_titles(v) for k, v in node.items() if k != "title"}
    if isinstance(node, list):
        return [_drop_titles(item) for item in node]
    return node


def _struct(model: type, *, keep_required: bool = True) -> dict[str, Any]:
    schema = _drop_titles(_drop_nulls(_inline_refs(model.model_json_schema())))
    if not keep_required:
        schema.pop("required", None)
    return schema


def build_input_schema() -> dict[str, Any]:
    from cell_performance_batch.batch import MAX_WORKERS_CAP
    from cell_performance_batch.cell_performance import (
        CellParametersInput,
        SimulationParameters,
    )

    cell_parameters = _struct(CellParametersInput)
    cell_parameters["description"] = (
        "Cell design: geometry, electrode formulation, foils, separator, "
        "electrolyte and voltage limits."
    )
    protocol = _struct(SimulationParameters)

    design = {
        "type": "object",
        "required": ["cell_parameters"],
        "properties": {
            "id": {
                "type": "string",
                "description": (
                    "Optional identifier echoed back on the matching result."
                ),
            },
            "name": {
                "type": "string",
                "description": "Optional human-readable label for the design.",
            },
            "cell_parameters": cell_parameters,
            "simulation_parameters": {
                **protocol,
                "description": (
                    "Per-design test protocol. Replaces the batch-level one "
                    "entirely for this design — it is not merged into it."
                ),
            },
        },
    }

    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "title": "Cell design batch",
        "required": ["designs"],
        "properties": {
            "designs": {
                "type": "array",
                "description": (
                    "Cell designs to evaluate. Results come back in this order."
                ),
                "items": design,
            },
            "simulation_parameters": {
                **protocol,
                "description": (
                    "Test protocol shared by every design that doesn't "
                    "declare its own."
                ),
            },
            "fail_fast": {
                "type": "boolean",
                "default": False,
                "description": (
                    "Stop at the first failing design; the rest report "
                    "status 'skipped'."
                ),
            },
            "max_workers": {
                "type": "integer",
                "default": 1,
                "minimum": 1,
                "maximum": MAX_WORKERS_CAP,
                "description": (
                    "Designs evaluated in parallel processes. >1 disables "
                    "progress and fail_fast."
                ),
            },
            "result_detail": {
                "type": "string",
                "default": "full",
                "enum": ["full", "kpis", "summary"],
                "description": (
                    "full = every field; kpis = drop timeseries/experiments/"
                    "BOM; summary = comparison scalars only."
                ),
            },
        },
    }


def build_output_schema() -> dict[str, Any]:
    from cell_performance_batch.batch import COMPARISON_KEYS
    from cell_performance_batch.cell_performance import CellPerformanceKPIs

    # ``required`` is dropped: under result_detail 'kpis' / 'summary' the
    # trimmed fields are genuinely absent, so promising them would make
    # the schema lie about two of the three modes.
    kpis = _struct(CellPerformanceKPIs, keep_required=False)
    kpis["description"] = (
        "Cell performance KPIs for this design. Null when status is not "
        "'ok'; a subset when result_detail trims the response."
    )

    comparison_row = {
        "type": "object",
        "properties": {
            "index": {"type": "integer"},
            "id": {"type": ["string", "null"]},
            "name": {"type": ["string", "null"]},
            "status": {"type": "string", "enum": ["ok", "error", "skipped"]},
            **{
                key: {
                    "type": ["number", "null"],
                    "description": _kpi_description(kpis, key),
                }
                for key in COMPARISON_KEYS
            },
        },
    }

    result = {
        "type": "object",
        "required": ["index", "status"],
        "properties": {
            "index": {
                "type": "integer",
                "description": "Position of this design in the input array.",
            },
            "id": {"type": ["string", "null"]},
            "name": {"type": ["string", "null"]},
            "status": {"type": "string", "enum": ["ok", "error", "skipped"]},
            "error": {
                "type": ["string", "null"],
                "description": (
                    "Failure message, or the upstream model's own error field."
                ),
            },
            "traceback": {"type": ["string", "null"]},
            "runtime_s": {"type": "number"},
            "kpis": {**kpis, "type": ["object", "null"]},
        },
    }

    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "title": "Cell design batch results",
        "required": ["results", "summary"],
        "properties": {
            "results": {
                "type": "array",
                "description": (
                    "One entry per input design, positionally aligned with "
                    "designs[]."
                ),
                "items": result,
            },
            "summary": {
                "type": "object",
                "description": (
                    "Batch-level counts plus a flat comparison table across " "designs."
                ),
                "properties": {
                    "design_count": {"type": "integer"},
                    "succeeded": {"type": "integer"},
                    "failed": {"type": "integer"},
                    "skipped": {"type": "integer"},
                    "runtime_s": {"type": "number"},
                    "result_detail": {"type": "string"},
                    "comparison": {
                        "type": "array",
                        "description": (
                            "Key scalar KPIs per design — chartable without "
                            "unpacking results[]."
                        ),
                        "items": comparison_row,
                    },
                },
            },
        },
    }


def _kpi_description(kpis: dict[str, Any], key: str) -> str:
    field = kpis.get("properties", {}).get(key, {})
    return field.get("description", key)


# ── TOML rendering ───────────────────────────────────────────────────

_BARE_KEY = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")


def _key(name: str) -> str:
    if name and set(name) <= _BARE_KEY:
        return name
    return json.dumps(name)


def _scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return json.dumps(value)
    if isinstance(value, str):
        # ensure_ascii=False so "[mg/cm²]" stays readable — TOML is UTF-8.
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ", ".join(_scalar(item) for item in value) + "]"
    raise TypeError(f"Not a TOML scalar: {value!r}")


def _is_table(value: Any) -> bool:
    return isinstance(value, dict)


def _is_table_array(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(map(_is_table, value))


def _render_table(data: dict[str, Any], path: str) -> list[str]:
    """Emit ``data`` as ``[path]``, scalars first then nested tables.

    TOML requires a table's own key/value pairs to precede any of its
    sub-tables, so the two passes below are not a style choice.
    """
    lines: list[str] = []
    scalars = {k: v for k, v in data.items() if not _is_table(v)}
    tables = {k: v for k, v in data.items() if _is_table(v)}

    for name, value in scalars.items():
        if _is_table_array(value):
            continue
        if isinstance(value, list) and any(map(_is_table, value)):
            raise TypeError(f"Mixed scalar/table array at {path}.{name}: {value!r}")
        lines.append(f"{_key(name)} = {_scalar(value)}")

    for name, value in scalars.items():
        if not _is_table_array(value):
            continue
        for item in value:
            lines.append("")
            lines.append(f"[[{path}.{_key(name)}]]")
            lines.extend(_render_table(item, f"{path}.{_key(name)}"))

    for name, value in tables.items():
        child = f"{path}.{_key(name)}"
        lines.append("")
        lines.append(f"[{child}]")
        lines.extend(_render_table(value, child))

    return lines


def render() -> str:
    sections = [HEADER]
    for name, schema in (
        ("input_schema", build_input_schema()),
        ("output_schema", build_output_schema()),
    ):
        path = f"wrapper.{name}"
        sections.append(f"\n[{path}]\n" + "\n".join(_render_table(schema, path)) + "\n")
    rendered = "".join(sections)

    # Round-trip guard: the emitter is bespoke, so never ship output we
    # haven't parsed back into the exact dicts we meant to write.
    parsed = tomllib.loads(rendered)
    for name, schema in (
        ("input_schema", build_input_schema()),
        ("output_schema", build_output_schema()),
    ):
        if parsed["wrapper"][name] != schema:
            raise AssertionError(f"{name} did not survive the TOML round trip.")
    return rendered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit nonzero if protos.toml differs from freshly generated output.",
    )
    args = parser.parse_args(argv)

    rendered = render()
    if args.check:
        if TOML_PATH.read_text() != rendered:
            print(
                "protos.toml is stale — run python scripts/generate_protos_toml.py",
                file=sys.stderr,
            )
            return 1
        print("protos.toml is up to date.")
        return 0

    TOML_PATH.write_text(rendered)
    print(f"Wrote {TOML_PATH} ({len(rendered):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
