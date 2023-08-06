#!/usr/bin/env bash
HERE=$(dirname "$0")
VERSION=${1:-"stable"}
REPO=${2:-"https://github.com/awslabs/gluonts.git"}
PKG=${3:-"gluonts"}
if [[ "$VERSION" == "latest" ]]; then
    VERSION="main"
fi

. "${HERE}/../shared/setup.sh" "${HERE}" true

if [[ "$VERSION" == "stable" ]]; then
    PIP install --no-cache-dir -U ${PKG}[torch]
elif [[ "$VERSION" =~ ^[0-9] ]]; then
    PIP install --no-cache-dir -U ${PKG}[torch]==${VERSION}
else
    TARGET_DIR="${HERE}/lib/${PKG}"
    rm -Rf ${TARGET_DIR}
    git clone --depth 1 --single-branch --branch ${VERSION} --recurse-submodules ${REPO} ${TARGET_DIR}
    PIP install -U -e ${TARGET_DIR}[torch]
fi

PY -c "from gluonts import __version__; print(__version__)" >> "${HERE}/.setup/installed"
