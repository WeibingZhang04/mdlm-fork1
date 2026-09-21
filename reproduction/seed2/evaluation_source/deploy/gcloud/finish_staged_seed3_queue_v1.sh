#!/usr/bin/env bash
set -euo pipefail
readonly EXPERIMENT_ROOT=/mnt/contextual-forest/staged-debug-20260907
readonly CODE_DIR="${1:?supply the authenticated analysis checkout}"
cd "${CODE_DIR}"
# This is the continuation of one training job, not an independent sweep.
# The VM's absolute 04:49:24 UTC shutdown timer remains authoritative.
while systemctl is-active --quiet staged-fresh-seed3-replication.service; do
  sleep 5
done
test -f "${EXPERIMENT_ROOT}/fresh-replication-seed3-retry1-v1/hashes.json"
/mnt/contextual-forest/venv/bin/python -u deploy/gcloud/finish_staged_fresh_replication_v1.py \
  --experiment-root "${EXPERIMENT_ROOT}" --single-seed 3 \
  > "${EXPERIMENT_ROOT}/logs/fresh-test-seed3-v1.log" 2>&1
tar --exclude='*/dev-cache' --exclude='*/dev-cache/*' \
  -czf /tmp/staged-fresh-seed3-complete-v1.tar.gz -C "${EXPERIMENT_ROOT}" \
  fresh-replication-seed3-retry1-v1 fresh-selection-seed3-v1 fresh-test-seed3-v1 \
  logs/fresh-replication-seed3-retry1-v1.log logs/fresh-test-seed3-v1.log
sha256sum /tmp/staged-fresh-seed3-complete-v1.tar.gz
