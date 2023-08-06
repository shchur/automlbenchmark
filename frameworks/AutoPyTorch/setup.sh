#!/usr/bin/env bash
HERE=$(dirname "$0")
VERSION=${1:-"stable"}
REPO=${2:-"https://github.com/automl/Auto-PyTorch.git"}
PKG=${3:-"autoPyTorch"}
if [[ "$VERSION" == "latest" ]]; then
    VERSION="main"
fi

. "${HERE}/../shared/setup.sh" "${HERE}" true

SUDO apt-get update -y
SUDO apt-get upgrade -y
SUDO apt-get install -y python3-opencv ffmpeg libsm6 libxext6
# As of 2023-08-07, AutoPyTorch-Forecsating installed from PyPI does not worked because of broken dependencies
# We pin the dependency versions using requirements.txt to fix this problem
PIP install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu
PIP install -U -r ${HERE}/requirements.txt
# Fix a bug where logger cannot be initialized because of a race condition
PIP install git+https://github.com/shchur/Auto-PyTorch.git@fix-logger-bug


PY -c "from autoPyTorch import __version__; print(__version__)" >> "${HERE}/.setup/installed"
