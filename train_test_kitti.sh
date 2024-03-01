#!/bin/zsh

DIR="/ubc/cs/research/kmyi/wsun/scratch/dataset/kitti360"

SCANS=("00" "01" "02" "03" "04")
ITERS=("5000" "10000" "15000" "20000" "25000" "30000")

for scan in ${SCANS[@]}; do
    python train.py -s $DIR \
        -m ./exp/kitti_full/train_$scan \
        -i $DIR/data_2d_nvs_drop50/train_$scan.txt \
        --data-device cpu \
        --iterations 30000 \
        --eval \
        --is_kitti_test

    for iter in ${ITERS[@]}; do
        python render.py -s $DIR \
            -m ./exp/kitti_full/train_$scan \
            -i $DIR/data_2d_nvs_drop50/train_$scan.txt \
            --data-device cpu \
            --iteration $iter \
            --is_kitti_test


        python /jeffyct-ssd/projects/GS-Collection/evaluate.py \
            -i ./exp/kitti/train_$scan/train/ours_$iter/renders \
            -g ./exp/kitti/train_$scan/train/ours_$iter/gt \
            -m psnr ssim lpips vgglpips \
            --prefix "train_${scan}_${iter}_train" | tee -a kitti.csv

        # python /jeffyct-ssd/projects/GS-Collection/evaluate.py \
        #     -i ./exp/kitti/train_$scan/test/ours_$iter/renders \
        #     -g ./exp/kitti/train_$scan/test/ours_$iter/gt \
        #     -m psnr ssim lpips vgglpips \
        #     --prefix "train_${scan}_${iter}_test" | tee -a kitti.csv
    done
done
