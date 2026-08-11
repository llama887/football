#!/usr/bin/env bash
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/$USER}
TOOLCHAIN_PREFIX=${TOOLCHAIN_PREFIX:-$SCRATCH_ROOT/.conda/envs/football-fast}
VENV=${VENV:-$REPO/.venv}
export CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS:-$SCRATCH_ROOT/.cache/conda/pkgs}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$SCRATCH_ROOT/.cache}

command -v uv >/dev/null || {
  echo "uv is required: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
}

mkdir -p "$CONDA_PKGS_DIRS" "$XDG_CACHE_HOME"
if [ ! -x "$TOOLCHAIN_PREFIX/bin/python" ]; then
  echo "Missing validated Torch environment: $TOOLCHAIN_PREFIX" >&2
  echo "Set TOOLCHAIN_PREFIX to the football-fast environment." >&2
  exit 1
fi

"$TOOLCHAIN_PREFIX/bin/python" -c \
  'import importlib.util as u; assert all(u.find_spec(x) for x in ("gym", "gymnasium", "pufferlib", "torch"))' \
  || {
    echo "$TOOLCHAIN_PREFIX must contain Gym, Gymnasium, PufferLib, and Torch." >&2
    exit 1
  }

module purge
export PATH="$TOOLCHAIN_PREFIX/bin:$PATH"
export CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DPython_ROOT_DIR=$VENV -DPython_FIND_VIRTUALENV=ONLY -DCMAKE_NO_SYSTEM_FROM_IMPORTED=ON"
export CC="$TOOLCHAIN_PREFIX/bin/x86_64-conda-linux-gnu-cc"
export CXX="$TOOLCHAIN_PREFIX/bin/x86_64-conda-linux-gnu-c++"
export CMAKE_PREFIX_PATH="$TOOLCHAIN_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export CPATH="$TOOLCHAIN_PREFIX/include${CPATH:+:$CPATH}"
export PKG_CONFIG_PATH="$TOOLCHAIN_PREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export LD_LIBRARY_PATH="$TOOLCHAIN_PREFIX/lib:$VENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-${BUILD_JOBS:-4}}

if [ ! -x "$VENV/bin/python" ]; then
  uv venv --system-site-packages --python "$TOOLCHAIN_PREFIX/bin/python" "$VENV"
fi

SITE_PACKAGES=$("$VENV/bin/python" -c 'import site; print(site.getsitepackages()[0])')
printf '%s\n%s\n' "$REPO" "$REPO/third_party" \
  > "$SITE_PACKAGES/puffer_football.pth"

(cd / && "$VENV/bin/python" -c \
  'import gfootball, importlib.util as u; assert all(u.find_spec(x) for x in ("gym", "gymnasium", "pufferlib", "torch"))')

echo "UV environment ready: $VENV"
