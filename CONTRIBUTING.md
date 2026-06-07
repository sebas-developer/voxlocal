# Contributing

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Checks

```bash
ruff check .
pytest
python -m build
```

Tests must not require downloaded models, network access, an audio device, or a
developer-specific cache.
