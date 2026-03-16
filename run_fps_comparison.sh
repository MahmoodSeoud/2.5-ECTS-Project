#!/bin/bash
#SBATCH --job-name=fps_compare
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

source /home/mseo/2.5-ECTS-Project/.venv/bin/activate
set -a; source /home/mseo/2.5-ECTS-Project/.env; set +a

cd /home/mseo/2.5-ECTS-Project
python run_pipeline_fps_comparison.py

echo "=== Job finished at $(date) ==="
