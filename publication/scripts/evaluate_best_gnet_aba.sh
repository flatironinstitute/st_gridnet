#!/bin/sh
#SBATCH -o /mnt/home/adaly/st_gridnet/publication/data/eval_gnet_aba.out
#SBATCH -e /mnt/home/adaly/st_gridnet/publication/data/eval_gnet_aba.err
#SBATCH --job-name=eval_gnet_aba

#SBATCH -p gpu
#SBATCH --gres=gpu:v100-32gb:1
#SBATCH -c 15
#SBATCH --mem=300GiB

module load slurm
module load gcc cuda cudnn python3

cd /mnt/home/adaly/st_gridnet/publication/scripts
python3 evaluate_best_models.py aba