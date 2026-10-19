# Releasing

Nothing is published until a GitHub release is published. Then
`.github/workflows/release.yml` runs the tests, checks the tag against the
package version and uploads the package to PyPI.

The docs are not deployed to GitHub Pages: a Pages site for this repository
would take over https://rohankraghu.github.io/honest-backtest/, which is the
write-up page on the personal site. They are read from the `docs` folder on
GitHub, and `mkdocs serve` shows them locally.

## One-time setup

1. **PyPI.** Sign in at pypi.org, open *Your account > Publishing > Add a new
   pending publisher*, and enter: project name `honest-backtester`, owner
   `rohanKraghu`, repository `honest-backtest`, workflow `release.yml`,
   environment `pypi`. This lets the workflow upload without an API token.
2. **GitHub environments.** In the repository's *Settings > Environments*,
   create `pypi` (optionally with yourself as a required reviewer, so every
   upload waits for your click).

## Each release

1. Set `__version__` in `src/honest_backtest/__init__.py` and move the
   changelog's "unreleased" heading to that version.
2. Check locally:

   ```bash
   pip install -e ".[dev,docs]"
   pytest
   python -m build && python -m twine check --strict dist/*
   mkdocs build --strict
   ```

3. Merge to `main`, then create a GitHub release whose tag is `v` plus the
   version (for example `v0.1.0`). Publishing the release starts the upload.

A version number can be uploaded to PyPI only once, even if it is later
deleted, so a mistake means a new version rather than a re-upload.
