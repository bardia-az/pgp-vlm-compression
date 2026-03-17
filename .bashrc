# Names
export PACKAGE_NAME="pgp-vlm-compression"
export SLACK_USER=""

# Versions
export VERSION_PYTHON="3.11.5"
export VERSION_POETRY="2.1.3"
export VERSION_PIP="24.2"
export VERSION_UBUNTU="24.04"

# Paths
export PATH="${PWD}/bin:${PATH}"
export PROJECT_DIR="${PWD}"
export POETRY_CACHE="${HOME}/.cache"
export DATA_PATH="${HOME}/data"
export LMUData="${PWD}/datasets"
export CLUSTER_PROJECT_DIR=""

# URIs
export DOCKER_URL=""
export PYPI_HOST=""

# Docker
if [[ -n "${USE_DOCKER}" ]]; then
	export DOCKERIZED_CMD="dockerized"
	export BUILDER_IMAGE_TASK="builder-image"
fi

export BUILDER_IMAGE="${DOCKER_URL}:${PACKAGE_NAME}_builder"
export TRAINING_IMAGE="${DOCKER_URL}:${PACKAGE_NAME}_training"
export TRAINING_BASE_IMAGE="ubuntu:${VERSION_UBUNTU}"

# Flags
export DEVELOPMENT_MODE=1

# Additional Credentials
export HF_TOKEN="${HF_TOKEN:-$(ansible-vault-view .huggingface.token)}"
