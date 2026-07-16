#!/bin/bash  -l
#SBATCH -J 'EquilibriumFitting'
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -o outLogEquilibrium
#SBATCH -e errLogEquilibrium
#SBATCH --mail-user=jamesjlin@ucsb.edu
#SBATCH --mail-type=ALL

module purge all
module load anaconda
source activate crm

cd $SLURM_SUBMIT_DIR

python run_equilibrium.py