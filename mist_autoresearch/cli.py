"""Command-line entry point for mist_autoresearch."""

import argparse
from pathlib import Path

from mist_autoresearch.postprocessing.researcher import PostprocessingResearcher
from mist_autoresearch.stopping import StoppingCriteria


def _add_stopping_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=50,
        help="Hard stop after this many iterations. (default: 50)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Stop early after this many consecutive iterations with no improvement. (default: 10)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold for the Wilcoxon stopping gate. (default: 0.05)",
    )
    parser.add_argument(
        "--min-iterations",
        type=int,
        default=5,
        help="Minimum iterations before early stopping is considered. (default: 5)",
    )
    parser.add_argument(
        "--min-patients-for-significance",
        type=int,
        default=15,
        help=(
            "Skip the significance gate if the dataset has fewer patients than "
            "this. (default: 15)"
        ),
    )


def _build_stopping(ns: argparse.Namespace) -> StoppingCriteria:
    return StoppingCriteria(
        max_iterations=ns.max_iterations,
        patience=ns.patience,
        alpha=ns.alpha,
        min_iterations=ns.min_iterations,
        min_patients_for_significance=ns.min_patients_for_significance,
    )


def _run_postprocessing(ns: argparse.Namespace) -> None:
    stopping = _build_stopping(ns)
    researcher = PostprocessingResearcher(
        config=Path(ns.config),
        predictions_dir=Path(ns.predictions),
        test_csv=Path(ns.test_csv),
        output_dir=Path(ns.output),
        stopping=stopping,
        model=ns.model,
        num_workers=ns.num_workers,
        additional_prompt=Path(ns.additional_prompt) if ns.additional_prompt else None,
    )
    researcher.run()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mist_autoresearch",
        description="Autoresearch tools for MIST medical image segmentation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    pp = subparsers.add_parser(
        "postprocessing",
        help="Automatically search for the best postprocessing strategy.",
        description=(
            "LLM-driven sequential search for a postprocessing strategy that "
            "improves segmentation quality. Each iteration proposes a strategy "
            "via the Claude Code CLI, evaluates it with mist_postprocess, and "
            "ranks it against all previous strategies. The loop stops when a "
            "patience or significance criterion is met, or when max_iterations "
            "is reached."
        ),
    )
    pp.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config.json from mist_analyze.",
    )
    pp.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Directory of baseline NIfTI predictions from mist_predict.",
    )
    pp.add_argument(
        "--test-csv",
        type=str,
        required=True,
        help="CSV with 'id' and 'mask' columns pointing to ground truth masks.",
    )
    pp.add_argument(
        "--output",
        type=str,
        required=True,
        help="Root output directory for the run.",
    )
    pp.add_argument(
        "--additional-prompt",
        type=str,
        default=None,
        metavar="FILE",
        help=(
            "Path to a Markdown file injected into every proposal prompt as "
            "'## Additional Context'. Use it to share dataset knowledge, "
            "evaluation criteria, or transform suggestions with the agent."
        ),
    )
    pp.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name forwarded to 'claude --model'. Defaults to Claude Code's active model.",
    )
    pp.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel workers for postprocessing and evaluation. (default: 1)",
    )
    _add_stopping_args(pp)

    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for the mist_autoresearch CLI."""
    parser = _build_parser()
    ns = parser.parse_args(argv)
    if ns.command == "postprocessing":
        _run_postprocessing(ns)


if __name__ == "__main__":
    main()  # pragma: no cover
