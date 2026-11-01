#!/bin/bash
#SBATCH --job-name=sa_casscf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_l40s
#SBATCH --output=sa_casscf.%j.out
#SBATCH --error=sa_casscf.%j.err

source /util/opt/anaconda/4.9.2/etc/profile.d/conda.sh
conda activate /mnt/nrdstor/centurion/kluong3/gpu4pyscf_env
module load cuda

export LD_LIBRARY_PATH=/mnt/nrdstor/centurion/kluong3/gpu4pyscf_env/lib/python3.12/site-packages/cutensor/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/mnt/nrdstor/centurion/kluong3/software/gpu4pyscf:${PYTHONPATH:-}

# Run the standalone GPU SA-CASSCF test
python -u run_sa_casscf.py
