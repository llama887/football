#!/bin/bash
#SBATCH --job-name=football-fast-test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --account=torch_pr_1066_tandon_advanced
#SBATCH --chdir=/scratch/fyy2003/repos/football
#SBATCH --output=/scratch/fyy2003/repos/football/sbatch/logs/%x-%j.out
#SBATCH --error=/scratch/fyy2003/repos/football/sbatch/logs/%x-%j.err

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export CMAKE_POLICY_VERSION_MINIMUM=3.5
export CONDA_PKGS_DIRS=/scratch/fyy2003/.cache/conda/pkgs
export XDG_CACHE_HOME=/scratch/fyy2003/.cache

ENV_PREFIX=/scratch/fyy2003/.conda/envs/football-fast
mkdir -p "$CONDA_PKGS_DIRS" "$XDG_CACHE_HOME" sbatch/logs

module purge
module load anaconda3/2025.06

if [ ! -x "$ENV_PREFIX/bin/python" ]; then
  conda create -y -p "$ENV_PREFIX" --override-channels -c conda-forge \
    python=3.10 cmake ninja cxx-compiler pkg-config boost boost-cpp \
    sdl2 sdl2_image sdl2_ttf sdl2_gfx libglu libgl libegl \
    libgl-devel libglx-devel libopengl-devel libegl-devel pip
fi

if [ ! -f "$ENV_PREFIX/include/GL/gl.h" ] || \
   [ ! -f "$ENV_PREFIX/include/EGL/egl.h" ]; then
  conda install -y -p "$ENV_PREFIX" --override-channels -c conda-forge \
    libgl-devel libglx-devel libopengl-devel libegl-devel
fi

module purge
export PATH="$ENV_PREFIX/bin:$PATH"
export CONDA_PREFIX="$ENV_PREFIX"
export CMAKE_ARGS="-DPython_ROOT_DIR=$ENV_PREFIX -DPython_FIND_VIRTUALENV=ONLY -DCMAKE_NO_SYSTEM_FROM_IMPORTED=ON"
export CC="$ENV_PREFIX/bin/x86_64-conda-linux-gnu-cc"
export CXX="$ENV_PREFIX/bin/x86_64-conda-linux-gnu-c++"
export CMAKE_PREFIX_PATH="$ENV_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export CPATH="$ENV_PREFIX/include${CPATH:+:$CPATH}"
export PKG_CONFIG_PATH="$ENV_PREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

"$ENV_PREFIX/bin/python" -m pip install \
  'pip<24' 'setuptools==65.5.0' 'wheel==0.38.4'
"$ENV_PREFIX/bin/python" -m pip install --no-build-isolation 'gym==0.21.0'
"$ENV_PREFIX/bin/python" -m pip install psutil 'gymnasium>=0.29.1' \
  'pufferlib>=3.0,<3.1'
if [ ! -e gfootball_engine ] && [ ! -L gfootball_engine ]; then
  ln -s third_party/gfootball_engine gfootball_engine
fi
if [ ! -e third_party/gfootball_engine/fonts ] && \
   [ ! -L third_party/gfootball_engine/fonts ]; then
  ln -s ../fonts third_party/gfootball_engine/fonts
fi
if [ -f third_party/gfootball_engine/Makefile ]; then
  make -C third_party/gfootball_engine clean
fi
gfootball/build_game_engine.sh
"$ENV_PREFIX/bin/python" -m gfootball.env.football_env_test \
  FootballEnvTest.test_fast_mode_matches_baseline_dynamics0 \
  FootballEnvTest.test_fast_mode_matches_baseline_dynamics1 \
  FootballEnvTest.test_fast_mode_matches_baseline_dynamics2
"$ENV_PREFIX/bin/python" -m gfootball.env.puffer_env_test
"$ENV_PREFIX/bin/python" -m gfootball.examples.benchmark_fast_mode \
  --steps=500 --repeats=3
