#!/bin/sh
#SBATCH -o /mnt/home/adaly/st_gridnet/publication/data/maynard_nested_xval/eval_maynard_xval.%A.out
#SBATCH -e /mnt/home/adaly/st_gridnet/publication/data/maynard_nested_xval/eval_maynard_xval.%A.err
#SBATCH --job-name=eval_maynard_xval

#SBATCH -p gpu
#SBATCH --gres=gpu:v100-32gb:1
#SBATCH -c 15
#SBATCH --mem=300GiB

module load slurm
module load gcc cuda cudnn python3

cd /mnt/home/adaly/st_gridnet/publication/scripts
python3 eval_maynard_xval.py