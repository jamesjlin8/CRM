#!/bin/bash  -l
#SBATCH -J '4cyl_17r_400l_nmax10'
#SBATCH --nodes=1
#SBATCH --ntasks=1  
#SBATCH --cpus-per-task=40
#SBATCH -o outLogFlowCalc4cyl_17r_400l_nmax10
#SBATCH --mail-user=jamesjlin@ucsb.edu
#SBATCH --mail-type=ALL

module purge all
module load anaconda
source activate crm

cd $SLURM_SUBMIT_DIR
export OMP_NUM_THREADS=1

python run_flowcalc.py --n-cyl 4 --radius 17 --length 400 --n-cyl-max 6