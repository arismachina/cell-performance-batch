"""Batch layer — run the vendored cell performance model over N designs.

The upstream model is strictly one-design-in / one-KPI-set-out. This
module wraps it in the array shape a design sweep actually wants:

    {"designs": [d0, d1, ...]}  ->  {"results": [r0, r1, ...], ...}

Two properties matter more than anything else here:

- **Positional integrity.** ``results[i]`` always describes
  ``designs[i]``, whatever happened to the other designs. Nothing is
  dropped, reordered, or filtered out of the array.
- **Failure isolation.** A design that fails validation or blows up
  inside PyBaMM produces an error entry, not a dead batch. Sweeps are
  exactly where one bad point is normal.

``pybamm`` is imported lazily (first design run), so a zero-design
batch — which is what the Protos build pipeline sends as its dry-run
sample input — stays fast and small enough for the 512 MB dry-run
container.
"""

from __future__ import annotations

import logging
import time
import traceback
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)

#: Compact per-design fields lifted into ``summary.comparison`` so a
#: sweep is readable (and chartable) without unpacking every result.
COMPARISON_KEYS: tuple[str, ...] = (
    "cell_nominal_capacity_Ah",
    "cell_nominal_energy_Wh",
    "cell_nominal_gravimetric_energy_density_Wh_kg",
    "cell_nominal_volumetric_energy_density_Wh_L",
    "cell_nominal_max_discharge_power_300s_W",
    "cell_dcir_10s_mohm",
    "cell_theoretical_capacity_Ah",
    "cell_theoretical_energy_Wh",
    "cell_mass_g",
    "cell_volume_L",
    "cell_n_p_ratio",
    "ocv100_V",
    "ocv0_V",
)

#: Dropped from each result under ``result_detail="kpis"``. These are
#: the unbounded payloads — a 40-design sweep carrying full timeseries
#: is tens of MB of JSON on stdout.
_BULK_KEYS: tuple[str, ...] = (
    "c3_discharge_timeseries",
    "experiment_results",
    "bill_of_materials",
)

#: Upper bound on ``max_workers``. The Protos K8s runner caps model
#: pods at 2 CPU / 4 GiB, and each PyBaMM process is memory-hungry, so
#: fanning out further trades throughput for OOM kills.
MAX_WORKERS_CAP = 4


class DesignSpec(BaseModel):
    """One cell design in the batch.

    ``cell_parameters`` / ``simulation_parameters`` are passed through
    to the upstream model untouched — they are validated there, by
    ``CellPerformanceInput``, so this layer never has to track upstream
    schema changes.
    """

    model_config = ConfigDict(extra="forbid")

    id: str | None = Field(
        None, description="Caller-supplied identifier echoed back on the result."
    )
    name: str | None = Field(None, description="Human-readable label for the design.")
    cell_parameters: dict[str, Any] = Field(
        ..., description="Upstream CellParametersInput payload."
    )
    simulation_parameters: dict[str, Any] | None = Field(
        None,
        description=(
            "Upstream SimulationParameters payload. When omitted, the "
            "batch-level simulation_parameters are used for this design."
        ),
    )


class BatchInput(BaseModel):
    """Array-in envelope."""

    model_config = ConfigDict(extra="forbid")

    designs: list[DesignSpec] = Field(
        default_factory=list, description="Designs to evaluate, in order."
    )
    simulation_parameters: dict[str, Any] | None = Field(
        None,
        description=(
            "Test protocol shared by every design that doesn't carry its "
            "own. Used verbatim — a per-design block replaces it whole "
            "rather than merging into it, so a design is always run "
            "under exactly one protocol you can point at."
        ),
    )
    fail_fast: bool = Field(
        False,
        description=(
            "Stop at the first failing design instead of evaluating the "
            "rest. Remaining designs are reported with status 'skipped'."
        ),
    )
    max_workers: int = Field(
        1,
        ge=1,
        description=(
            f"Designs to evaluate in parallel processes (capped at "
            f"{MAX_WORKERS_CAP}). 1 runs in-process, which is the only "
            "mode that reports progress."
        ),
    )
    result_detail: Literal["full", "kpis", "summary"] = Field(
        "full",
        description=(
            "'full' = every upstream field; 'kpis' = drop timeseries, "
            "experiment results and the bill of materials; 'summary' = "
            "only the comparison scalars."
        ),
    )

    @model_validator(mode="after")
    def _require_a_protocol(self) -> BatchInput:
        missing = [
            index
            for index, design in enumerate(self.designs)
            if design.simulation_parameters is None
        ]
        if missing and self.simulation_parameters is None:
            raise ValueError(
                f"designs at index {missing} have no simulation_parameters "
                "and no batch-level simulation_parameters to fall back on."
            )
        return self


def run_batch(
    payload: dict[str, Any],
    *,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    """Evaluate every design in ``payload`` and return the array result.

    ``payload`` is the raw ``BatchInput`` dict (wire boundary, same
    convention as the upstream model's ``calculate_cell_performance``).
    ``progress_callback(pct, message)`` is optional and only fires in
    the sequential path — a process pool has no ordered progress to
    report.

    Raises ``pydantic.ValidationError`` when the envelope itself is
    malformed. Per-design failures never raise; they land in the
    corresponding result entry.
    """
    parsed = BatchInput.model_validate(payload)
    started = time.monotonic()

    if not parsed.designs:
        return _envelope([], parsed, elapsed_s=time.monotonic() - started)

    workers = min(parsed.max_workers, MAX_WORKERS_CAP, len(parsed.designs))
    if workers > 1:
        results = _run_pooled(parsed, workers)
    else:
        results = _run_sequential(parsed, progress_callback)

    return _envelope(results, parsed, elapsed_s=time.monotonic() - started)


def _run_sequential(
    parsed: BatchInput,
    progress_callback: Any | None,
) -> list[dict[str, Any]]:
    total = len(parsed.designs)
    results: list[dict[str, Any]] = []
    stop_after: int | None = None

    for index, design in enumerate(parsed.designs):
        if stop_after is not None:
            results.append(_skipped(index, design, stop_after))
            continue
        if progress_callback is not None:
            progress_callback(
                int(100 * index / total),
                f"Evaluating design {index + 1}/{total}",
            )
        result = _run_one(index, design, parsed)
        results.append(result)
        if parsed.fail_fast and result["status"] == "error":
            stop_after = index

    if progress_callback is not None:
        progress_callback(100, f"Evaluated {total} design(s)")
    return results


def _run_pooled(parsed: BatchInput, workers: int) -> list[dict[str, Any]]:
    """Parallel path — no progress, no fail_fast.

    ``fail_fast`` is deliberately ignored here rather than
    approximated: with N designs already in flight there is no
    well-defined "first" failure to stop at, and pretending otherwise
    would make the skipped set depend on scheduling order.
    """
    from concurrent.futures import ProcessPoolExecutor

    if parsed.fail_fast:
        logger.warning("fail_fast is ignored when max_workers > 1 — every design runs.")

    payloads = [
        (
            index,
            design.model_dump(),
            _protocol_for(design, parsed),
            parsed.result_detail,
        )
        for index, design in enumerate(parsed.designs)
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_run_one_unpacked, payloads))


def _run_one_unpacked(
    args: tuple[int, dict[str, Any], dict[str, Any], str],
) -> dict[str, Any]:
    """Process-pool entry point — must be module-level to be picklable."""
    index, design_dict, protocol, result_detail = args
    design = DesignSpec.model_validate(design_dict)
    return _evaluate(index, design, protocol, result_detail=result_detail)


def _run_one(
    index: int,
    design: DesignSpec,
    parsed: BatchInput,
) -> dict[str, Any]:
    return _evaluate(
        index,
        design,
        _protocol_for(design, parsed),
        result_detail=parsed.result_detail,
    )


def _evaluate(
    index: int,
    design: DesignSpec,
    protocol: dict[str, Any],
    *,
    result_detail: str,
) -> dict[str, Any]:
    """Run one design. Never raises — failures become error entries."""
    # Lazy: importing the model pulls in PyBaMM (hundreds of MB of
    # resident memory), which a zero-design batch must not pay for.
    from cell_performance_batch.cell_performance import calculate_cell_performance

    started = time.monotonic()
    try:
        kpis = calculate_cell_performance(
            {
                "cell_parameters": design.cell_parameters,
                "simulation_parameters": protocol,
            }
        )
    except Exception as exc:  # noqa: BLE001 - one bad design must not kill the sweep
        logger.exception("design %d (%s) failed", index, design.name or design.id)
        return {
            "index": index,
            "id": design.id,
            "name": design.name,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
            "runtime_s": round(time.monotonic() - started, 3),
            "kpis": None,
        }

    if not kpis:
        # The upstream model returns {} when its internal computation
        # returns None — a failure it swallows rather than raises.
        return {
            "index": index,
            "id": design.id,
            "name": design.name,
            "status": "error",
            "error": "Model returned no KPIs (upstream computation failed).",
            "traceback": None,
            "runtime_s": round(time.monotonic() - started, 3),
            "kpis": None,
        }

    return {
        "index": index,
        "id": design.id,
        "name": design.name,
        "status": "ok",
        "error": kpis.get("error"),
        "traceback": None,
        "runtime_s": round(time.monotonic() - started, 3),
        "kpis": _trim(kpis, result_detail),
    }


def _skipped(index: int, design: DesignSpec, failed_at: int) -> dict[str, Any]:
    return {
        "index": index,
        "id": design.id,
        "name": design.name,
        "status": "skipped",
        "error": f"Skipped: fail_fast stopped the batch at design {failed_at}.",
        "traceback": None,
        "runtime_s": 0.0,
        "kpis": None,
    }


def _protocol_for(design: DesignSpec, parsed: BatchInput) -> dict[str, Any]:
    """Per-design protocol wins whole; otherwise the batch-level one."""
    if design.simulation_parameters is not None:
        return design.simulation_parameters
    # Validated in BatchInput._require_a_protocol.
    return parsed.simulation_parameters or {}


def _trim(kpis: dict[str, Any], result_detail: str) -> dict[str, Any]:
    if result_detail == "full":
        return kpis
    if result_detail == "kpis":
        return {k: v for k, v in kpis.items() if k not in _BULK_KEYS}
    return {k: kpis.get(k) for k in COMPARISON_KEYS}


def _envelope(
    results: list[dict[str, Any]],
    parsed: BatchInput,
    *,
    elapsed_s: float,
) -> dict[str, Any]:
    counts = {"ok": 0, "error": 0, "skipped": 0}
    for result in results:
        counts[result["status"]] += 1

    comparison = [
        {
            "index": result["index"],
            "id": result["id"],
            "name": result["name"],
            "status": result["status"],
            **{key: (result["kpis"] or {}).get(key) for key in COMPARISON_KEYS},
        }
        for result in results
    ]

    return {
        "results": results,
        "summary": {
            "design_count": len(results),
            "succeeded": counts["ok"],
            "failed": counts["error"],
            "skipped": counts["skipped"],
            "runtime_s": round(elapsed_s, 3),
            "result_detail": parsed.result_detail,
            "comparison": comparison,
        },
    }
