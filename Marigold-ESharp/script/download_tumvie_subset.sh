#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:-/path/to/data/event3d/TUM-VIE/subset}
mkdir -p "$ROOT"

download_one() {
  local seq="$1"
  local url="$2"
  mkdir -p "$ROOT/$seq"
  wget -c --show-progress -P "$ROOT/$seq" "$url"
}

# Calibration files only; no full calibration sequences.
wget -c --show-progress -P "$ROOT" https://tumevent-vi.vision.in.tum.de/camera-calibrationA.json
wget -c --show-progress -P "$ROOT" https://tumevent-vi.vision.in.tum.de/camera-calibrationB.json
wget -c --show-progress -P "$ROOT" https://tumevent-vi.vision.in.tum.de/mocap-imu-calibrationA.json
wget -c --show-progress -P "$ROOT" https://tumevent-vi.vision.in.tum.de/mocap-imu-calibrationB.json

# Small representative subset: left events + grayscale/image/IMU/GT data only. No right events.
download_one mocap-desk2 https://tumevent-vi.vision.in.tum.de/mocap-desk2/mocap-desk2-events_left.h5
download_one mocap-desk2 https://tumevent-vi.vision.in.tum.de/mocap-desk2/mocap-desk2-vi_gt_data.tar.gz

download_one mocap-6dof https://tumevent-vi.vision.in.tum.de/mocap-6dof/mocap-6dof-events_left.h5
download_one mocap-6dof https://tumevent-vi.vision.in.tum.de/mocap-6dof/mocap-6dof-vi_gt_data.tar.gz

download_one running-easy https://tumevent-vi.vision.in.tum.de/running-easy/running-easy-events_left.h5
download_one running-easy https://tumevent-vi.vision.in.tum.de/running-easy/running-easy-vi_gt_data.tar.gz

download_one floor2-dark https://tumevent-vi.vision.in.tum.de/floor2-dark/floor2-dark-events_left.h5
download_one floor2-dark https://tumevent-vi.vision.in.tum.de/floor2-dark/floor2-dark-vi_gt_data.tar.gz

echo "Downloaded subset to $ROOT"
echo "Extract vi data with: find $ROOT -name '*vi_gt_data.tar.gz' -execdir tar -xzf '{}' ';'"
