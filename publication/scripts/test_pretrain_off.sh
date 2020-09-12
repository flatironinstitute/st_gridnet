#!/bin/sh
#SBATCH -o /mnt/home/adaly/st_gridnet/publication/data/rnet_rinit.%A_%a.out
#SBATCH -e /mnt/home/adaly/st_gridnet/publication/data/rnet_rinit.%A_%a.err
#SBATCH --job-name=rnet_rinit

#SBATCH --array=0-4

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 15
#SBATCH --mem=300GiB

module load slurm
module load gcc cuda cudnn python3

cd /mnt/home/adaly/st_gridnet/publication/scripts
python3 test_pretrain.py