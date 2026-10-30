"""Command line interface: ``flotation-benchmark fetch | run | report``."""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_DATA = Path("data/raw")
DEFAULT_RAW = DEFAULT_DATA / "quality_prediction_mining.parquet"
DEFAULT_RESULTS = Path("results")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="flotation-benchmark", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("fetch", help="download the raw data from OpenML")
    p.add_argument("--dest", type=Path, default=DEFAULT_DATA)
    p.add_argument("--force", action="store_true", help="download again even if present")

    p = sub.add_parser("run", help="run all protocols and models")
    p.add_argument("--raw", type=Path, default=DEFAULT_RAW,
                   help="raw file (OpenML parquet or Kaggle CSV)")
    p.add_argument("--out", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--gap-hours", type=int, default=24,
                   help="embargo between train and test in walk-forward CV")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--protocols", nargs="*", help="subset of protocol keys")
    p.add_argument("--models", nargs="*", help="subset of model names")
    p.add_argument("--quick", action="store_true",
                   help="use every 10th 20 s row and skip permutation importances")

    p = sub.add_parser("report", help="build the HTML report from saved results")
    p.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--out", type=Path, default=None,
                   help="output HTML file (default: <results>/report.html)")

    args = parser.parse_args(argv)

    if args.cmd == "fetch":
        from .data import fetch

        fetch(args.dest, force=args.force)
    elif args.cmd == "run":
        from .benchmark import PROTOCOL_BY_KEY, Config, run
        from .data import RawDataError
        from .models import ALL_MODELS

        if not args.raw.exists():
            parser.error(f"{args.raw} not found; run `flotation-benchmark fetch` first")
        bad = sorted(set(args.protocols or []) - set(PROTOCOL_BY_KEY))
        if bad:
            parser.error(f"unknown protocol(s): {', '.join(bad)}; "
                         f"choose from {', '.join(PROTOCOL_BY_KEY)}")
        bad = sorted(set(args.models or []) - set(ALL_MODELS))
        if bad:
            parser.error(f"unknown model(s): {', '.join(bad)}; choose from {', '.join(ALL_MODELS)}")
        cfg = Config(
            n_folds=args.folds, gap_hours=args.gap_hours, seed=args.seed, n_jobs=args.jobs,
            rows_stride=10 if args.quick else 1,
            importance_repeats=0 if args.quick else 5,
        )
        try:
            run(args.raw, args.out, cfg, protocols=args.protocols, models=args.models or ALL_MODELS)
        except RawDataError as exc:
            hint = ("; try `flotation-benchmark fetch --force`"
                    if args.raw == DEFAULT_RAW else "")
            parser.error(f"{exc}{hint}")
        print(f"results written to {args.out}")
    elif args.cmd == "report":
        from .report import build_report

        if not (args.results / "summary.csv").exists():
            parser.error(f"no results in {args.results}; run `flotation-benchmark run` first")
        out = build_report(args.results, args.out)
        print(f"report written to {out}")
    return 0
