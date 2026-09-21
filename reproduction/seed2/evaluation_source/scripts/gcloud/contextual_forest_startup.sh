#!/usr/bin/env bash
set -euo pipefail

readonly MOUNT_POINT='/mnt/contextual-forest'
readonly HARD_STOP_UNIT='contextual-forest-hard-stop'

# Shut down immediately when a prerequisite fails before the timer is known to
# be active.  Startup-script failure alone does not stop a Compute Engine VM.
fail_and_shutdown() {
  local message="$1"
  local status="${2:-1}"
  echo "${message}" >&2
  /usr/sbin/shutdown -h now || true
  exit "${status}"
}

# Install the compute guard before metadata lookup or disk work.  This still
# shuts the guest down if metadata is unavailable or mounting fails during
# provisioning.
# An independently configured account-level spending guard is recommended.
systemctl stop "${HARD_STOP_UNIT}.timer" 2>/dev/null || true
systemctl reset-failed "${HARD_STOP_UNIT}.service" 2>/dev/null || true
if ! systemd-run \
    --unit="${HARD_STOP_UNIT}" \
    --on-active=4h \
    --timer-property=AccuracySec=1min \
    /usr/sbin/shutdown -h now; then
  fail_and_shutdown 'Failed to install the four-hour compute guard'
fi

metadata_disk=''
if [[ -z "${CONTEXTUAL_FOREST_DATA_DISK_NAME:-}" ]]; then
  if ! metadata_disk="$(
      curl --fail --silent --show-error \
        --connect-timeout 2 \
        --max-time 5 \
        -H 'Metadata-Flavor: Google' \
        'http://metadata.google.internal/computeMetadata/v1/instance/attributes/contextual-forest-data-disk'
    )"; then
    fail_and_shutdown \
      'Could not read contextual-forest-data-disk instance metadata' 2
  fi
fi
readonly DATA_DISK_NAME="${CONTEXTUAL_FOREST_DATA_DISK_NAME:-${metadata_disk}}"
if [[ -z "${DATA_DISK_NAME}" ]]; then
  fail_and_shutdown 'Missing contextual-forest-data-disk instance metadata' 2
fi
readonly DATA_DEVICE="/dev/disk/by-id/google-${DATA_DISK_NAME}"

mkdir -p "${MOUNT_POINT}"
for _ in $(seq 1 60); do
  if [[ -b "${DATA_DEVICE}" ]]; then
    break
  fi
  sleep 2
done
if [[ ! -b "${DATA_DEVICE}" ]]; then
  fail_and_shutdown "Dedicated data disk ${DATA_DEVICE} did not appear"
fi

filesystem_type="$(lsblk -no FSTYPE "${DATA_DEVICE}" | tr -d '[:space:]')"
if [[ -z "${filesystem_type}" ]]; then
  mkfs.ext4 -F -m 0 -L contextual-forest-data "${DATA_DEVICE}"
elif [[ "${filesystem_type}" != 'ext4' ]]; then
  fail_and_shutdown \
    "Refusing unexpected filesystem ${filesystem_type} on ${DATA_DEVICE}"
fi

filesystem_uuid="$(blkid -s UUID -o value "${DATA_DEVICE}")"
fstab_entry="UUID=${filesystem_uuid} ${MOUNT_POINT} ext4 defaults,nofail 0 2"
if ! grep -q "UUID=${filesystem_uuid}" /etc/fstab; then
  printf '%s\n' "${fstab_entry}" >> /etc/fstab
fi
mountpoint -q "${MOUNT_POINT}" || mount "${MOUNT_POINT}"
chmod 0777 "${MOUNT_POINT}"

date --iso-8601=seconds > "${MOUNT_POINT}/last-boot.txt"
nvidia-smi > "${MOUNT_POINT}/nvidia-smi-at-boot.txt" 2>&1 || true
