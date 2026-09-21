#!/usr/bin/env bash
set -euo pipefail

# This worker uses a newly cloned data disk. Never format or run old queues.
readonly DATA_DEVICE=/dev/disk/by-id/google-diffusion-staged-debug-data-20260907
readonly MOUNT_POINT=/mnt/contextual-forest
for attempt in $(seq 1 60); do
  [[ -b "${DATA_DEVICE}" ]] && break
  sleep 1
done
[[ -b "${DATA_DEVICE}" ]]
blkid "${DATA_DEVICE}"
mkdir -p "${MOUNT_POINT}"
mountpoint -q "${MOUNT_POINT}" || mount "${DATA_DEVICE}" "${MOUNT_POINT}"

# A second, guest-side hard stop bounds cost even if no experiment is launched.
systemd-run --unit=staged-debug-hardstop --on-active=4h /sbin/shutdown -h now
nvidia-smi
