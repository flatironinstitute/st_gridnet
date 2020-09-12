#!/bin/sh
#SBATCH -o /mnt/home/adaly/st_gridnet/publication/data/gnhex_maynard.%A_%a.out
#SBATCH -e /mnt/home/adaly/st_gridnet/publication/data/gnhex_maynard.%A_%a.err
#SBATCH --job-name=gnhex_maynard

#SBATCH --array=0-9

#SBATCH -p gpu
#SBATCH --gres=gpu:v100-32gb:1
#SBATCH -c 15
#SBATCH --mem=300GiB

module load slurm
module load gcc cuda cudnn python3

cd /mnt/home/adaly/st_gridnet/publication/scripts
python3 fit_hex.py