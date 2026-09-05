"""Protos wrapper — MODEL_INPUT (JSON) in, one JSON line on stdout.

Contract (docs/WRAPPER_SPEC.md in protos-v2):
  1. Read JSON from the MODEL_INPUT environment variable.
  2. Write exactly one JSON line to stdout; the last line wins.
  3. Diagnostics go to stderr — surfaced to the user on failure.
  4. Exit 0 on success, nonzero on failure.
  5. Only /tmp is writable; /app/repo is read-only; no network.

The heavy lifting is in the installed package, not here — the build
pip-installs the repo, so ``import cell_performance_batch`` resolves.
The sys.path fallback covers the case where that install step was
skipped (e.g. running this file straight out of a clone).
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

# Fallback for an uninstalled checkout: <repo>/src on the path.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _progress(pct: int, message: str) -> None:
    """Progress goes to stderr — stdout is reserved for the result."""
    print(f"[{pct:3d}%] {message}", file=sys.stderr, flush=True)


def _json_safe(value: Any) -> Any:
    """Replace non-finite floats with null, recursively.

    A diverging solve can leave NaN/Inf in a KPI. ``json.dumps``
    happily writes those as bare ``NaN``/``Infinity`` tokens, which is
    not valid JSON — the consumer downstream would fail to parse the
    whole batch over one bad number.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def main() -> int:
    raw = os.environ.get("MODEL_INPUT")
    if raw is None:
        print("MODEL_INPUT is not set.", file=sys.stderr)
        return 1

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"MODEL_INPUT is not valid JSON: {exc}", file=sys.stderr)
        return 1

    if not isinstance(payload, dict):
        print(
            f"MODEL_INPUT must be a JSON object, got {type(payload).__name__}.",
            file=sys.stderr,
        )
        return 1

    from cell_performance_batch import run_batch

    try:
        # The vendored model prints its own step log ("[3/5] Running
        # C/3 discharge ...") to stdout. Harmless under the "last line
        # wins" rule, but keeping stdout to exactly one JSON line means
        # a consumer that reads the whole stream still parses.
        with contextlib.redirect_stdout(sys.stderr):
            result = run_batch(payload, progress_callback=_progress)
    except Exception as exc:  # noqa: BLE001 - envelope-level failure
        # Per-design failures are captured inside the result array;
        # reaching here means the envelope itself was unusable.
        import traceback

        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        return 1

    print(json.dumps(_json_safe(result), allow_nan=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
