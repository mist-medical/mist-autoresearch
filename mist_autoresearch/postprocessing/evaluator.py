"""Evaluator for the postprocessing autoresearch loop."""
import json
import subprocess
from pathlib import Path

import pandas as pd


class PostprocessingEvaluator:
    """Runs mist_postprocess (with inline evaluation) for one strategy.

    Wraps the ``mist_postprocess`` CLI so the orchestrator never needs to
    import MIST internals directly for file I/O or postprocessing.

    Args:
        predictions_dir: Directory of baseline NIfTI predictions from
            ``mist_predict``.
        test_csv: CSV with ``id`` and ``mask`` columns pointing to ground
            truth masks.
        config: Path to ``config.json`` produced by ``mist_analyze``.
        num_workers: Worker count forwarded to ``--num-workers-postprocess``
            and ``--num-workers-evaluate``.
    """

    def __init__(
        self,
        predictions_dir: Path,
        test_csv: Path,
        config: Path,
        num_workers: int = 1,
    ) -> None:
        self.predictions_dir = Path(predictions_dir)
        self.test_csv = Path(test_csv)
        self.config = Path(config)
        self.num_workers = num_workers

    def run(self, strategy: list, output_dir: Path) -> pd.DataFrame:
        """Apply *strategy* to predictions and return per-patient results.

        Writes ``strategy.json`` to *output_dir*, calls ``mist_postprocess``
        (which writes predictions + ``postprocess_results.csv`` into
        *output_dir*), and returns the results DataFrame.

        Args:
            strategy: List of strategy steps (empty list = no transforms).
            output_dir: Directory for this iteration's outputs.

        Returns:
            DataFrame with ``id`` and metric columns.

        Raises:
            subprocess.CalledProcessError: If ``mist_postprocess`` exits
                with a non-zero status.
            FileNotFoundError: If the results CSV is missing after the run.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        strategy_path = output_dir / "strategy.json"
        strategy_path.write_text(json.dumps(strategy, indent=2))

        cmd = [
            "mist_postprocess",
            "--base-predictions", str(self.predictions_dir),
            "--output", str(output_dir),
            "--postprocess-strategy", str(strategy_path),
            "--paths-csv", str(self.test_csv),
            "--eval-config", str(self.config),
            "--num-workers-postprocess", str(self.num_workers),
            "--num-workers-evaluate", str(self.num_workers),
        ]
        subprocess.run(cmd, check=True)

        results_csv = output_dir / "postprocess_results.csv"
        if not results_csv.exists():
            raise FileNotFoundError(
                f"mist_postprocess did not produce results at {results_csv}"
            )
        return pd.read_csv(results_csv)
