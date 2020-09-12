#!/bin/sh
#SBATCH -o /mnt/home/adaly/st_gridnet/publication/data/gnhex_maynard_loo.%A_%a.out
#SBATCH -e /mnt/home/adaly/st_gridnet/publication/data/gnhex_maynard_loo.%A_%a.err
#SBATCH --job-name=gnhex_maynard_loo

#SBATCH --array=0-11

#SBATCH -p gpu
#SBATCH --gres=gpu:v100-32gb:1
#SBATCH -c 15
#SBATCH --mem=300GiB

module load slurm
module load gcc cuda cudnn python3

cd /mnt/home/adaly/st_gridnet/publication/scripts
python3 fit_hex_loo.py $SLURM_ARRAY_TASK_ID 0.0006374 0.09602