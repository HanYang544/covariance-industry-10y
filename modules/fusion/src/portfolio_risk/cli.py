"""Command-line entry point for the experiment pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

STAGES = (
    "prepare-data",
    "run-baselines",
    "train",
    "evaluate",
    "backtest",
    "run-preliminary",
    "run-window-ablation",
    "run-feature-search",
    "run-forward-audit",
    "run-state-research",
    "run-online-research",
    "run-evidence-audit",
    "run-confirmation",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="portfolio-risk",
        description="Run multiscale portfolio risk forecasting experiments.",
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)
    for stage in STAGES:
        command = subparsers.add_parser(stage)
        default = "configs/preliminary.yaml" if stage == "run-preliminary" else "configs/base.yaml"
        if stage == "run-window-ablation":
            default = "configs/window_ablation.yaml"
        if stage in ("run-feature-search", "run-forward-audit"):
            default = "configs/feature_search.yaml"
        if stage == "run-state-research":
            default = "configs/state_whitener.yaml"
        if stage == "run-online-research":
            default = "configs/online_whitener.yaml"
        if stage == "run-evidence-audit":
            default = "configs/evidence_audit.yaml"
        if stage == "run-confirmation":
            default = "configs/confirmation.yaml"
        command.add_argument("--config", type=Path, default=Path(default))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.config.is_file():
        raise SystemExit(f"Configuration file not found: {args.config}")
    if args.stage == "run-preliminary":
        from portfolio_risk.experiment import run_preliminary

        run_preliminary(args.config)
    elif args.stage == "run-window-ablation":
        from portfolio_risk.window_ablation import run_window_ablation

        run_window_ablation(args.config)
    elif args.stage == "run-feature-search":
        from portfolio_risk.feature_search import run_feature_search

        run_feature_search(args.config)
    elif args.stage == "run-forward-audit":
        from portfolio_risk.forward_audit import run_forward_audit

        run_forward_audit(args.config)
    elif args.stage == "run-state-research":
        from portfolio_risk.state_research import run_state_research

        run_state_research(args.config)
    elif args.stage == "run-online-research":
        from portfolio_risk.online_research import run_online_research

        run_online_research(args.config)
    elif args.stage == "run-evidence-audit":
        from portfolio_risk.evidence_audit import run_evidence_audit

        run_evidence_audit(args.config)
    elif args.stage == "run-confirmation":
        from portfolio_risk.confirmation import run_confirmation

        run_confirmation(args.config)
    else:
        raise SystemExit(
            f"Standalone stage '{args.stage}' is not implemented yet. "
            "Use run-preliminary --config configs/preliminary.yaml for the complete first round."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

