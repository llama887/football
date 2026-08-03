#!/bin/bash
#SBATCH --job-name=football-regularized
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=48:00:00
#SBATCH --account=torch_pr_1066_tandon_advanced
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=fyy2003@nyu.edu
#SBATCH --chdir=/scratch/fyy2003/repos/football
#SBATCH --output=/scratch/fyy2003/repos/football/sbatch/logs/%x-%j.out
#SBATCH --error=/scratch/fyy2003/repos/football/sbatch/logs/%x-%j.err

set -euo pipefail

REPO=/scratch/fyy2003/repos/football
ENV_PREFIX=/scratch/fyy2003/.conda/envs/football-fast
RUN_ROOT=/scratch/fyy2003/experiments/football-regularized-${SLURM_JOB_ID}
mkdir -p "$RUN_ROOT/checkpoints" "$REPO/sbatch/logs"

module purge
export PATH="$ENV_PREFIX/bin:$PATH"
export CONDA_PREFIX="$ENV_PREFIX"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$REPO:$REPO/third_party"
export GFOOTBALL_DATA_DIR="$REPO/third_party/gfootball_engine/data"
export GFOOTBALL_FONT="$REPO/third_party/fonts/AlegreyaSansSC-ExtraBold.ttf"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

if [ -f /scratch/fyy2003/repos/Puffer-Soccer/.env ]; then
  set -a
  source /scratch/fyy2003/repos/Puffer-Soccer/.env
  set +a
fi

export GPU_HEARTBEAT_TARGET_UTILIZATION=70
export GPU_HEARTBEAT_CHECK_INTERVAL=0.2
export GPU_HEARTBEAT_MATRIX_SIZE=6144
export GPU_HEARTBEAT_DTYPE=bfloat16

cleanup() {
  if [ -n "${HEARTBEAT_PID:-}" ] && kill -0 "$HEARTBEAT_PID" 2>/dev/null; then
    kill "$HEARTBEAT_PID"
    wait "$HEARTBEAT_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

nvidia-smi
nice -n 19 /scratch/fyy2003/repos/Puffer-Soccer/.venv/bin/python -u \
  /scratch/fyy2003/repos/Puffer-Soccer/sbatch/gpu_heartbeat.py &
HEARTBEAT_PID=$!
echo "Started GPU heartbeat with PID: $HEARTBEAT_PID"

cd "$RUN_ROOT"
"$ENV_PREFIX/bin/python" -u -m gfootball.examples.train_puffer \
  --device cuda \
  --num-workers 30 \
  --frame-stack 4 \
  --curriculum-levels 11 \
  --curriculum-window 20 \
  --curriculum-success-threshold 0.6 \
  --total-timesteps 1000000000 \
  --wandb \
  --wandb-project google-football-fast-rl \
  --wandb-group regularized-self-play \
  --data-dir "$RUN_ROOT/checkpoints"
