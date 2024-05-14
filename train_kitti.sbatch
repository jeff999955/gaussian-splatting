#!/bin/bash
#SBATCH --job-name=kitti
#SBATCH --output=exp/kitti/%a.log
#SBATCH --array=0-4
#SBATCH --partition=snubfin
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --mem=64000

mamba activate /ubc/cs/research/kmyi/jeffyct/gaussian_splat_env
which python

DIR="/ubc/cs/research/kmyi/wsun/scratch/dataset/kitti360"

SCANS=("00" "01" "02" "03" "04")
ITERS=("30000")

EXP_NAME="kitti_resample_0"

scan=${SCANS[$SLURM_ARRAY_TASK_ID]}
iter=${ITERS[0]} 

/usr/bin/env python train.py -s $DIR \
    -m ./exp/$EXP_NAME/train_$scan \
    -i $DIR/data_2d_nvs_drop50/train_$scan.txt \
    --iterations 30000 \
    --eval \
    --resample_pnts 0

/usr/bin/env python render.py -s $DIR \
    -m ./exp/$EXP_NAME/train_$scan \
    -i $DIR/data_2d_nvs_drop50/train_$scan.txt \
    --data-device cpu \
    --iteration $iter 

/usr/bin/env python /ubc/cs/research/kmyi/jeffyct/GS-Collection/evaluate.py \
    -i ./exp/$EXP_NAME/train_$scan/train/ours_$iter/renders \
    -g ./exp/$EXP_NAME/train_$scan/train/ours_$iter/gt \
    -m psnr ssim lpips vgglpips \
    --prefix "train_${scan}_${iter}_train" | tee -a $EXP_NAME.csv

/usr/bin/env python /ubc/cs/research/kmyi/jeffyct/GS-Collection/evaluate.py \
    -i ./exp/$EXP_NAME/train_$scan/test/ours_$iter/renders \
    -g ./exp/$EXP_NAME/train_$scan/test/ours_$iter/gt \
    -m psnr ssim lpips vgglpips \
    --prefix "train_${scan}_${iter}_test" | tee -a $EXP_NAME.csv
