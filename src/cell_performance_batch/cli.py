"""Local CLI — run a batch from a JSON file without a container.

    cell-performance-batch examples/two_designs.json -o results.json

Same code path the Protos wrapper takes, so a batch that works here
works there. Progress lines go to stderr; the result JSON goes to
stdout (or ``-o``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cell-performance-batch",
        description="Evaluate an array of cell designs.",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Batch input JSON file ('-' to read stdin).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write results here instead of stdout.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation for the result (default: 2).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-design progress on stderr.",
    )
    args = parser.parse_args(argv)

    raw = sys.stdin.read() if str(args.input) == "-" else args.input.read_text()
    payload = json.loads(raw)

    from cell_performance_batch.batch import run_batch

    def progress(pct: int, message: str) -> None:
        print(f"[{pct:3d}%] {message}", file=sys.stderr, flush=True)

    result = run_batch(payload, progress_callback=None if args.quiet else progress)
    rendered = json.dumps(result, indent=args.indent, default=str)

    if args.output is not None:
        args.output.write_text(rendered)
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(rendered)

    # Nonzero when nothing succeeded — makes the CLI usable in a script.
    return 0 if result["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
