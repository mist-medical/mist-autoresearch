# MIST Autoresearch

LLM-driven autoresearch tools for [MIST](https://github.com/mist-medical/MIST) medical image segmentation experiments.

## What it does

`mist_autoresearch` runs sequential research loops where Claude proposes experiment strategies, evaluates them against prior results, and iterates until a stopping criterion is met. Each run produces a `research_notebook.md` with the agent's step-by-step reasoning alongside evaluation metrics.

## Commands

| Command | Description |
|---|---|
| `mist_autoresearch postprocessing` | Automated search for the best postprocessing strategy |

## Install

```console
pip install -e .
```

Requires a working MIST installation and the Claude Code CLI (`claude`) installed and authenticated.

## Quick start

```console
mist_autoresearch postprocessing \
  --config results/config.json \
  --predictions predictions/test \
  --test-csv data/test.csv \
  --output autoresearch/postprocessing/run1
```

See [docs/getting_started.md](docs/getting_started.md) and [docs/usage.md](docs/usage.md) for full documentation.

## How it works

The postprocessing loop:

1. Evaluates baseline predictions (no postprocessing) to establish a reference score.
2. Asks Claude — via the Claude Code CLI (`claude -p`) — to propose a postprocessing strategy, returned as a JSON object (`steps` plus a written `narrative`) parsed from its response.
3. Applies the strategy with `mist_postprocess` and evaluates with `mist_evaluate`.
4. Ranks all strategies tried so far using `mist_rank` (BraTS-style mean rank).
5. Feeds the ranking and pairwise significance table back to Claude.
6. Repeats until `--max-iterations` is reached or early stopping fires (patience + Wilcoxon significance).

## Output

```
autoresearch/postprocessing/run1/
├── research_notebook.md       # Agent reasoning at each iteration
├── history.json               # Full resumable run log + stopping reason
├── summary.json               # Winning strategy, ready to feed to mist_postprocess
├── rankings.csv               # Cumulative strategy rankings, best first
├── significance.csv           # Pairwise Wilcoxon significance matrix
├── baseline/
└── iteration_001/ ...
```

`rankings.csv`, `summary.json`, and `history.json` always agree on which strategy won.

## Stopping criteria

- **Hard stop**: `--max-iterations` (default 50)
- **Patience**: No improvement for `--patience` consecutive iterations (default 10). An
  iteration counts as an improvement only when the strategy it proposed reaches the top
  of the cumulative ranking.
- **Significance gate**: Best strategy is significantly better than baseline (p < `--alpha`, default 0.05) — required alongside patience for early stopping, unless the dataset is too small for the Wilcoxon test or baseline is itself the best strategy

If patience runs out while a non-baseline strategy leads but is not yet significant, the
loop keeps searching to `--max-iterations`. See [docs/usage.md](docs/usage.md#stopping-logic).

## License

Apache 2.0
