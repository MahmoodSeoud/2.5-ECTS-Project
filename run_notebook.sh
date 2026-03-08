#!/bin/bash
#SBATCH --job-name=sam2_pose
#SBATCH --partition=acltr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

echo "=== Job started at $(date) ==="
echo "Node: $(hostname)"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Activate venv
source /home/mseo/2.5-ECTS-Project/.venv/bin/activate

# Load API key from .env
set -a; source /home/mseo/2.5-ECTS-Project/.env; set +a

# Run notebook from export/ dir so relative paths work
cd /home/mseo/2.5-ECTS-Project/export

jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=86400 \
    --ExecutePreprocessor.kernel_name=sam2-env \
    --output sam2_pose_comparison_executed.ipynb \
    sam2_pose_comparison.ipynb

echo "=== Job finished at $(date) ==="
