#!/bin/sh
#SBATCH -o /mnt/home/adaly/st_gridnet/publication/data/eval_rnseg_maniatis.out
#SBATCH -e /mnt/home/adaly/st_gridnet/publication/data/eval_rnseg_maniatis.err
#SBATCH --job-name=eval_rnseg_maniatis

#SBATCH -p gpu
#SBATCH --gres=gpu:v100-32gb:1
#SBATCH -c 15
#SBATCH --mem=300GiB

module load slurm
module load gcc cuda cudnn python3

cd /mnt/home/adaly/st_gridnet/publication/scripts
python3 evaluate_best_models.py maniatis --resnetseg