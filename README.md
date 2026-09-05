# cell-performance-batch

Evaluate **an array of cell designs** and get **an array of performance
KPI sets** back — one entry per design, in input order.

The physics is not new: `src/cell_performance_batch/cell_performance.py`
is a verbatim copy of the `cell_performance` model from the Aris Protos
monorepo (equilibrium KPIs from formulation + geometry, then a PyBaMM
SPMe calibration pipeline). That model is strictly one-design-in /
one-result-out. This repo adds the batch layer around it and packages
the whole thing as a Protos **Python cloud model**, so a design sweep is
one model run instead of N.

---

## The contract

**In** (`MODEL_INPUT`, or a JSON file for the CLI):

```json
{
  "designs": [
    {
      "id": "pouch-80ah-baseline",
      "name": "Baseline",
      "cell_parameters": { "form_factor": "Pouch", "...": "..." }
    },
    {
      "id": "pouch-80ah-high-loading",
      "cell_parameters": { "...": "..." },
      "simulation_parameters": { "...": "per-design override" }
    }
  ],
  "simulation_parameters": { "...": "shared test protocol" },
  "fail_fast": false,
  "max_workers": 1,
  "result_detail": "full"
}
```

| Field | Meaning |
|---|---|
| `designs[]` | Required. Each needs `cell_parameters`; `id` / `name` are optional and echoed back. |
| `designs[].simulation_parameters` | Optional. **Replaces** the batch-level protocol for that design — it is not merged into it, so every design runs under exactly one protocol you can point at. |
| `simulation_parameters` | The protocol for every design that doesn't declare its own. Required unless all of them do. |
| `fail_fast` | Stop at the first failure; the remaining designs come back as `status: "skipped"`. Ignored when `max_workers > 1`. |
| `max_workers` | Designs per parallel process, capped at 4. `1` (default) is the only mode that reports progress. |
| `result_detail` | `full` (everything) · `kpis` (drops timeseries, experiment results, BOM) · `summary` (comparison scalars only). |

`cell_parameters` and `simulation_parameters` are passed straight through
to the vendored model and validated there — same fields, same units, same
meaning as the single-design model. See
[`docs/CELL_PERFORMANCE_MODEL.md`](docs/CELL_PERFORMANCE_MODEL.md) for
the field reference and `examples/two_designs.json` for a working payload.

Both structs are **declared in full** in `protos.toml`, so Protos sees
all 64 cell-design fields, all 35 simulation-protocol fields (per design
and batch-level) and the complete KPI output struct — units, bounds,
enums, nested material/experiment objects and all. They are generated
from the vendored Pydantic models rather than hand-written; see
[Schema generation](#schema-generation).

**Out** (one JSON line on stdout):

```json
{
  "results": [
    {
      "index": 0,
      "id": "pouch-80ah-baseline",
      "name": "Baseline",
      "status": "ok",
      "error": null,
      "traceback": null,
      "runtime_s": 1.42,
      "kpis": { "cell_nominal_capacity_Ah": 67.3, "...": "..." }
    }
  ],
  "summary": {
    "design_count": 2,
    "succeeded": 1,
    "failed": 1,
    "skipped": 0,
    "runtime_s": 2.89,
    "result_detail": "full",
    "comparison": [{ "index": 0, "cell_nominal_capacity_Ah": 67.3, "...": "..." }]
  }
}
```

Two guarantees the batch layer exists to provide:

- **Positional integrity.** `results[i]` describes `designs[i]`, always.
  Nothing is dropped, filtered, or reordered — including failures.
- **Failure isolation.** A design that fails validation or diverges in
  the solver yields `status: "error"` with the message, and the sweep
  carries on. One bad point in a sweep is normal, not fatal.

`summary.comparison` is a flat row per design carrying the headline
scalars (capacity, energy, densities, DCIR, mass, volume, N/P, OCV
limits) — chartable directly, without walking into `results[].kpis`.

---

## Registering it in Protos

The repo ships a [`protos.toml`](protos.toml), so Protos builds it
deterministically — no LLM wrapper drafting step.

1. Push this repo to GitHub (public, or private with your GitHub account
   connected in Protos).
2. In Protos: **Models Library → Register a model → GitHub**.
3. Paste the repo URL, optionally pin a branch / tag / commit, and hit
   analyse. Protos shallow-clones, reads `protos.toml`, and shows
   *Cell Performance Batch* with its declared input/output schemas.
4. Save. Protos builds the container (Kaniko), runs a dry-run with a
   generated sample input, and flips the model to **ready**.

The model then appears like any other in the Simulation Studio canvas
and to the Co-Engineer, with `designs` as an array input socket.

What the build does with this repo: base image `aris/base-python:3.12`,
`pip install /app/repo` (picked up from `pyproject.toml`, which is why
PyBaMM lands in the image), then `python /app/wrapper.py` per run with
the payload in `MODEL_INPUT`. No `build.extra_steps` are needed.

**A gotcha worth knowing:** `protos.toml` accepts
`input_schema_path` / `output_schema_path`, and the platform parses
those fields but never resolves the files they point at — a
path-declared schema silently ships as an empty schema and the Execute
Model UI degrades to a raw JSON textarea. This repo therefore declares
both schemas **inline**, and `tests/test_protos_contract.py` fails if
anyone switches them to the path form.

## Schema generation

`protos.toml` is ~3,200 lines because the full structs are inlined into
it; everything below `[wrapper.input_schema]` is generated:

```bash
python scripts/generate_protos_toml.py           # rewrite protos.toml
python scripts/generate_protos_toml.py --check   # CI-style staleness check
```

The generator reads `CellParametersInput`, `SimulationParameters` and
`CellPerformanceKPIs` from the vendored model and applies three
transforms on the way to TOML:

- **`$ref` / `$defs` are inlined.** Nothing in the platform's
  `protos.toml` path resolves references, so a surviving `$ref` is an
  opaque hole in the schema. Cycles raise rather than recurse.
- **Nulls and auto-titles are dropped.** TOML has no null (this only
  ever hits `"default": null`), and Pydantic's titleised field names
  mangle units — "Positive Electrode Mass Loading Mg Cm2" against a
  description that already reads "Positive mass loading [mg/cm²]".
- **`required` is dropped inside `kpis`.** Under `result_detail` of
  `kpis` or `summary` those fields are genuinely absent, so requiring
  them would make the schema lie about two of the three modes.

The emitter is bespoke, so it round-trips its own output through
`tomllib` and refuses to write anything that doesn't parse back to the
dicts it meant to emit. `tests/test_protos_contract.py` then checks the
committed file against a fresh render, field-for-field against the
Pydantic models, and reproduces the platform's dry-run sample generator
to confirm the generated schema still yields a runnable sample. Edit the
models, re-run the generator, commit both.

### Runtime limits to design around

The Protos model runner caps a run at **2 CPU / 4 GiB / 30 minutes**.
A design with `perform_rpt` plus multi-cycle experiments can take
minutes on its own, so size batches accordingly: start with
`result_detail: "kpis"` and a handful of designs, and split very large
sweeps into several runs rather than raising `max_workers` (parallel
processes multiply the memory, and the pod OOMs before it speeds up).

---

## Local use

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run the bundled two-design sweep
cell-performance-batch examples/two_designs.json -o results.json

# Or exactly what the container does
MODEL_INPUT="$(cat examples/two_designs.json)" python protos/wrapper.py
```

From Python:

```python
from cell_performance_batch import run_batch

result = run_batch({"designs": [...], "simulation_parameters": {...}})
```

Impedance steps (`EISStepConfig`) additionally need `pip install
".[eis]"`; every other path works without it.

## Tests

```bash
pytest                  # everything (the slow tests run a real solve)
pytest -m "not slow"    # contract-only, no PyBaMM run
```

`tests/test_batch.py` covers the array contract — alignment, isolation,
`fail_fast`, protocol precedence, `result_detail`, the parallel path.
`tests/test_protos_contract.py` covers the platform-facing surface:
`protos.toml` validity, the wrapper's `MODEL_INPUT` handling, exit
codes, one-JSON-line-on-stdout, and NaN sanitising.

## Keeping the model in sync with Protos

`src/cell_performance_batch/cell_performance.py` is vendored, not
authored here. Keep it byte-identical to upstream — that is what makes
these results comparable to the built-in `cell_performance` model.

To re-vendor:

```bash
cp <protos-v2>/backend/app/domains/models/cell_performance.py \
   src/cell_performance_batch/cell_performance.py
# re-apply the "VENDORED" header note, then:
# update UPSTREAM_COMMIT in src/cell_performance_batch/__init__.py
pytest
```

Current vendored revision: `b5339610e0e44df648aa394e27165e68f5400ee2`
(protos-v2). `ruff` is configured to skip that file so a re-vendor stays
a copy rather than a merge conflict.
