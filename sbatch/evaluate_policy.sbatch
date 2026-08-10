#!/bin/bash
#SBATCH --job-name=football-policy-eval
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --account=torch_pr_1066_tandon_advanced
#SBATCH --chdir=/scratch/fyy2003/repos/football
#SBATCH --output=/scratch/fyy2003/repos/football/sbatch/logs/%x-%j.out
#SBATCH --error=/scratch/fyy2003/repos/football/sbatch/logs/%x-%j.err

set -euo pipefail

CHECKPOINT=${1:?checkpoint path is required}
ATTACKER_ONLY_LEVELS=${ATTACKER_ONLY_LEVELS:-0}
REPO=/scratch/fyy2003/repos/football
ENV_PREFIX=/scratch/fyy2003/.conda/envs/football-fast
RUN_DIR=/scratch/fyy2003/experiments/football-policy-eval-${SLURM_JOB_ID}

module purge
export PATH="$ENV_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$REPO:$REPO/third_party"
export GFOOTBALL_DATA_DIR="$REPO/third_party/gfootball_engine/data"
export GFOOTBALL_FONT="$REPO/third_party/fonts/AlegreyaSansSC-ExtraBold.ttf"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

if [ -f /scratch/fyy2003/repos/Puffer-Soccer/.env ]; then
  set -a
  source /scratch/fyy2003/repos/Puffer-Soccer/.env
  set +a
fi

mkdir -p "$RUN_DIR"
cd "$RUN_DIR"
"$ENV_PREFIX/bin/python" -u -m gfootball.examples.evaluate_puffer_policy \
  --checkpoint "$CHECKPOINT" \
  --episodes 20 \
  --attacker-only-levels "$ATTACKER_ONLY_LEVELS" \
  --wandb-project google-football-fast-rl \
  --wandb-group policy-postmortem
