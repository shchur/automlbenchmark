#!/usr/bin/env bash
HERE=$(dirname "$0")
VERSION=${1:-"stable"}
REPO=${2:-"https://github.com/winedarksea/AutoTS.git"}
PKG=${3:-"autots"}
if [[ "$VERSION" == "latest" ]]; then
    VERSION="main"
fi

. "${HERE}/../shared/setup.sh" "${HERE}" true

# Taken from https://winedarksea.github.io/AutoTS/build/html/source/tutorial.html#safest-bet-for-installation
PIP install numpy scipy scikit-learn statsmodels lightgbm xgboost numexpr bottleneck yfinance pytrends fredapi --exists-action i
PIP install pystan prophet --exists-action i  # conda-forge option below works more easily, --no-deps to pip install prophet if this fails
PIP install tensorflow
PIP install mxnet --no-deps     # check the mxnet documentation for more install options, also try pip install mxnet --no-deps
PIP install gluonts arch
PIP install holidays-ext pmdarima dill greykite --exists-action i --no-deps
PIP install --upgrade numpy pandas --exists-action i  # mxnet likes to (pointlessly seeming) install old versions of numpy

if [[ "$VERSION" == "stable" ]]; then
    PIP install --no-cache-dir -U ${PKG}
elif [[ "$VERSION" =~ ^[0-9] ]]; then
    PIP install --no-cache-dir -U ${PKG}==${VERSION}
else
    TARGET_DIR="${HERE}/lib/${PKG}"
    rm -Rf ${TARGET_DIR}
    git clone --depth 1 --single-branch --branch ${VERSION} --recurse-submodules ${REPO} ${TARGET_DIR}
    PIP install -U -e ${TARGET_DIR}
fi


PY -c "from autots import __version__; print(__version__)" >> "${HERE}/.setup/installed"
