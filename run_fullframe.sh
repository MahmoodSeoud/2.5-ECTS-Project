#!/bin/bash
#SBATCH --job-name=yolo_fullframe
#SBATCH --partition=acltr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

echo "=== Job started at $(date) ==="
echo "Node: $(hostname)"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

source /home/mseo/2.5-ECTS-Project/.venv/bin/activate

cd /home/mseo/2.5-ECTS-Project
python run_yolo_fullframe.py

echo "=== Job finished at $(date) ==="
