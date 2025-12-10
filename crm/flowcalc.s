#!/bin/bash  -l
#SBATCH -J '5cyl_10r_100l'
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH -o outLogFlowCalc5cyl_10r_100l
#SBATCH -e errLogFlowCalc5cyl_10r_100l
#SBATCH --mail-user=jamesjlin@ucsb.edu
#SBATCH --mail-type=ALL

module purge all
module load anaconda
source activate crm

cd $SLURM_SUBMIT_DIR

export OMP_NUM_THREADS=1

python run_flowcalc.py