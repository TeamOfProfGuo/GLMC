#!/bin/bash

#SBATCH --job-name=Cifar_10_VS
#SBATCH --output=Cifar_10_VS.out
#SBATCH --error=Cifar_10_VS.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=40GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --requeue

singularity exec --nv --bind /scratch/hy2611 --overlay /scratch/hy2611/overlay-25GB-500K.ext3:ro /scratch/hy2611/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "
source /ext3/env.sh
cd /scratch/hy2611/GLMC
python VSmain.py --dataset cifar10 -a resnet32 --num_classes 10 --imbanlance_rate 0.01 --beta 0.5 --lr 0.01 --epochs 200 -b 64 --momentum 0.9 --weight_decay 5e-3 --resample_weighting 0.0 --label_weighting 1.2 --contrast_weight 4
"
