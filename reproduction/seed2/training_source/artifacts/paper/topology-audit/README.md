# Topology-audit paper summary

`summary.json` is the compact, paper-facing projection of the complete
14,400-record topology analysis. The full aggregate is intentionally kept out
of Git because it is 64 MiB; the summary binds both the aggregate's internal
analysis digest and the SHA-256 of the downloaded JSON file.

The source aggregate was deterministically replayed from all raw records with
`scripts/aggregate_topology_diagnostics.py` at repository revision
`7f161d3da3f059c5b310d165fa7fa5668d217616`. All values in the paper's
topology table and discussion are copied from the bound aggregate without
rounding until presentation.
