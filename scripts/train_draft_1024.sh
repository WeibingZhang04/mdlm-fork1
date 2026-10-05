#!/bin/bash
# Compatibility alias; train_four_models.sh is the requested main launcher.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/train_four_models.sh" "$@"
