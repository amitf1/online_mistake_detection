#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
REPO_URL="${REPO_URL:-https://github.com/amitf1/online_mistake_detection.git}"
REPO_DIR="${REPO_DIR:-${WORKSPACE_DIR}/online_mistake_detection}"
DATA_ZIP_URL="${DATA_ZIP_URL:-http://www.lsta.media.kyoto-u.ac.jp/resource/data/EgoOops/videos-processed-720p.zip}"
IMAGE_NAME="${IMAGE_NAME:-qwen-omd-dataloaders:latest}"
BUILD_DOCKER_IMAGE="${BUILD_DOCKER_IMAGE:-false}"

apt-get update
apt-get install -y --no-install-recommends git wget unzip byobu python3-pip ca-certificates
python3 -m pip install --upgrade pip || python3 -m pip install --break-system-packages --upgrade pip
python3 -m pip install --upgrade vastai wandb || python3 -m pip install --break-system-packages --upgrade vastai wandb

mkdir -p "${WORKSPACE_DIR}"
cd "${WORKSPACE_DIR}"

if [[ -d "${REPO_DIR}/.git" ]]; then
  git -C "${REPO_DIR}" pull --ff-only
else
  git clone "${REPO_URL}" "${REPO_DIR}"
fi

if [[ -n "${VAST_API_KEY:-}" ]]; then
  mkdir -p "${HOME}/.config/vastai"
  printf "%s\n" "${VAST_API_KEY}" > "${HOME}/.config/vastai/vast_api_key"
  chmod 600 "${HOME}/.config/vastai/vast_api_key"
fi

DATA_DIR="${REPO_DIR}/data"
VIDEO_DIR="${DATA_DIR}/videos-processed-720p"
ZIP_PATH="${DATA_DIR}/videos-processed-720p.zip"
mkdir -p "${DATA_DIR}"

if ! compgen -G "${VIDEO_DIR}"'/*/*.MP4' > /dev/null; then
  if [[ ! -f "${ZIP_PATH}" ]]; then
    wget -O "${ZIP_PATH}" "${DATA_ZIP_URL}"
  fi

  first_entry="$(python3 - "${ZIP_PATH}" <<'PY'
import sys
import zipfile

with zipfile.ZipFile(sys.argv[1]) as archive:
    names = [name for name in archive.namelist() if name and not name.endswith("/")]
print(names[0] if names else "")
PY
)"

  if [[ "${first_entry}" == videos-processed-720p/* ]]; then
    unzip -q -o "${ZIP_PATH}" -d "${DATA_DIR}"
  else
    mkdir -p "${VIDEO_DIR}"
    unzip -q -o "${ZIP_PATH}" -d "${VIDEO_DIR}"
  fi
fi

PROJECT_DIR="${REPO_DIR}/qwen_ego_oops_lora_dataloaders"
if [[ "${BUILD_DOCKER_IMAGE}" =~ ^(1|true|TRUE|yes|YES)$ ]]; then
  if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    docker build -t "${IMAGE_NAME}" "${PROJECT_DIR}"
  else
    echo "Docker is not available; skipping image build." >&2
  fi
fi

echo "Vast.ai setup complete."
echo "Repo: ${REPO_DIR}"
echo "Videos: ${VIDEO_DIR}"
echo "Project: ${PROJECT_DIR}"
echo "Next: copy ${REPO_DIR}/.env.example to ${REPO_DIR}/.env, fill secrets, then run:"
echo "  cd ${PROJECT_DIR} && bash scripts/run_module_a_vastai.sh"
