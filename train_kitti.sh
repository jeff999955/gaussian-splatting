#!/bin/zsh
if [ $# -lt 1 ]; then
    echo "Usage: $0 <scene_id>"
    exit 1
fi

DIR=`realpath $1`

SCANS=("00") #  "01" "02" "03" "04")
SCAN="${2:-train_00}"
MODEL_DIR="kitti_exp_train_00"

for scan in ${SCANS[@]}; do
    model_dir="kitti_exp_train_${scan}"
    python train.py -s $DIR \
        -m $DIR/$model_dir \
        -i $DIR/data_2d_nvs_drop50/train_$scan.txt \
        --data-device cpu \
        --iterations 30000 \
        --eval \
        --opacity_reset_interval 1000
done
