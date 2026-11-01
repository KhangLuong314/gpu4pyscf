#!/bin/bash -l
#SBATCH --job-name=bagel_ethylene
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=batch
#SBATCH --output=bagel.%j.out
#SBATCH --error=bagel.%j.err

export BAGEL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OMP_NUM_THREADS=1

module load anaconda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /mnt/nrdstor/centurion/kluong3/libra_env

export PATH="/mnt/nrdstor/centurion/kluong3/software/bagel/local_install:${PATH:-}"
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

# Run BAGEL on ethylene CAS(4,4) SA(3)-CASSCF
mpirun BAGEL ethylene_bagel.json > bagel_ethylene.out
