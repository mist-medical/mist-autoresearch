# Getting Started

## Requirements

- Python ≥ 3.10
- A working MIST installation (`pip install mist-medical`)
- Claude Code CLI installed and authenticated (`claude` available on your PATH)

## Install

```console
git clone https://github.com/mist-medical/mist-autoresearch.git
cd mist-autoresearch
pip install -e .
```

## Prerequisites

Before running `mist_autoresearch`, you need a completed MIST experiment:

| What you need              | Where it comes from     |
|----------------------------|-------------------------|
| `config.json`              | Output of `mist_analyze` |
| Predictions directory      | Output of `mist_predict` |
| Test CSV (`id`, `mask` columns) | Your data split    |

## Quick Start

```console
mist_autoresearch postprocessing \
  --config results/config.json \
  --predictions predictions/test \
  --test-csv data/test.csv \
  --output autoresearch/postprocessing/run1
```

This will:

1. Evaluate baseline predictions (no postprocessing) to establish a reference.
2. Ask Claude to propose a postprocessing strategy.
3. Apply the strategy with `mist_postprocess` and evaluate the result.
4. Rank all strategies tried so far against the baseline.
5. Feed the results back to Claude and repeat.
6. Stop when patience + significance criteria are met or `--max-iterations` is reached.
7. Write a `research_notebook.md` with the agent's reasoning at each step.

See [Usage](usage.md) for all available options.
