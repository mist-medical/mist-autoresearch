"""Persistent history for the autoresearch loop."""
import json
from datetime import datetime
from pathlib import Path


class History:
    """Tracks all iterations and the current best to support resumable runs.

    Writes to a JSON file after every update so the run can be resumed if
    interrupted.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._data: dict = self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> dict:
        if self.path.exists():
            return json.loads(self.path.read_text())
        return {
            "iterations": [],
            "best_iteration": None,
            "stopped_reason": None,
            "started_at": datetime.now().isoformat(),
        }

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2))

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def add_iteration(
        self,
        iteration: int,
        strategy: list,
        narrative: str,
        results_csv: str,
        mean_rank: float,
        p_value_vs_baseline: float | None,
    ) -> None:
        """Append an iteration record and flush to disk."""
        self._data["iterations"].append(
            {
                "iteration": iteration,
                "strategy": strategy,
                "narrative": narrative,
                "results_csv": results_csv,
                "mean_rank": mean_rank,
                "p_value_vs_baseline": p_value_vs_baseline,
                "timestamp": datetime.now().isoformat(),
            }
        )
        self._save()

    def update_best(self, iteration: int) -> None:
        """Mark an iteration as the current best and flush to disk."""
        self._data["best_iteration"] = iteration
        self._save()

    def set_stopped_reason(self, reason: str) -> None:
        """Record why the loop stopped and flush to disk."""
        self._data["stopped_reason"] = reason
        self._save()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def iterations(self) -> list[dict]:
        return self._data["iterations"]

    @property
    def best_iteration(self) -> int | None:
        return self._data["best_iteration"]

    @property
    def n_iterations(self) -> int:
        return len(self._data["iterations"])

    @property
    def stopped_reason(self) -> str | None:
        return self._data["stopped_reason"]
