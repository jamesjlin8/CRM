#!/bin/bash -l
#SBATCH -J '10modes3m'
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -o outLog10modes3m
#SBATCH -e errLog10modes3m
#SBATCH --mail-user=jamesjlin@ucsb.edu
#SBATCH --mail-type=ALL

module purge all
module load anaconda
source activate crm

cd $SLURM_SUBMIT_DIR

python run_pca_analysis.py --input-dir output --output-dir pca3m --q-min 0.008 --n-modes 10
python run_xg_model.py --pca-dir pca3m --models-dir 10modes3m