# Historical experiments

- `legacy/`: earlier drivers, single-agent baselines, loggers, and `agents/`.
- `results/`: earlier CSV/conversation logs, embedding cache, and single-agent database.

Files were moved without changing their contents. These are historical snapshots:
their imports, credentials, and relative data paths may require the original
layout to rerun. Some also depend on the current `agents2` package. Use Git history
to reproduce an exact earlier implementation and environment.

The maintained entry point is `driver3.py`; current results remain in
`query_log3.csv` at the repository root.
