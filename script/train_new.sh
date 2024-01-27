#!/bin/bash

#SBATCH --job-name=100_mx0a3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=80GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu
#SBATCH --mail-type=END
#SBATCH --mail-user=jx2314@nyu.edu
#SBATCH --partition=v100,rtx8000

#SBATCH --output=cf100resnet34ce_100_mx0a3.out

# job info
MIXUP=$1
ALPHA=$2
IMB=$3


source /scratch/$USER/GLMC/script/api.sh
# Singularity path
ext3_path=/scratch/$USER/overlay-25GB-500K.ext3
sif_path=/scratch/$USER/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif

# start running
# singularity exec --nv \
# --overlay ${ext3_path}:ro \
# ${sif_path} /bin/bash -c "
# source /ext3/env.sh
# python main_wb.py --dataset cifar10 -a resnet32 --imbanlance_rate 1 --beta 0.5 --lr 0.01 \
#  --epochs 200 --loss ce --resample_weighting 0 --mixup ${MIXUP} --mixup_alpha ${ALPHA} --store_name ce_mx${MIXUP}a${ALPHA}

echo "start"
singularity exec --nv \
    --overlay ${ext3_path}:ro \
    --overlay /scratch/lg154/sseg/dataset/tiny-imagenet-200.sqf:ro \
    ${sif_path} /bin/bash -c "
    source /ext3/env.sh
    conda activate GLMC
    python -m main_wb --dataset cifar100 -a resnet32 --imbanlance_rate ${IMB} --beta 0.5 --lr 0.01 \
    --epochs 200 --loss ce --resample_weighting 0 --mixup ${MIXUP} --mixup_alpha ${ALPHA} --store_name ce_mx${MIXUP}a${ALPHA} \
    --wandb_key ${wandb_key}"
