# `antmaxxing` Maintenance

## Setup

To set up the development environment, u need the following:

- [uv](https://github.com/astral-sh/uv)

After installing `uv`, run the following command to set up the environment:

```bash
uv sync
```

and run

```bash
pre-commit install
```

## Checks

As part of the pre-commit checks, `./scripts/check.sh` will be run. This script will check the following:

- `ruff` formatter
- `ruff` linter
- `mypy` type checker

You also can run quick fixes for `ruff` using `./scripts/fix.sh` script. This will run `ruff` formatter and linter with
`--fix` flag.