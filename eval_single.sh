#!/usr/bin/zsh
DIR="/jeffyct-ssd/nerf_synthetic"
MODEL_DIR="./nerf_synthetic_stride_colmap"

for scene in `ls $DIR`; do 
    if [ ! -d $DIR/$scene ]; then
        continue
    fi
    echo $scene
    for stride in 1 2 4 8; do
        python ~/Downloads/evaluate.py -i  $MODEL_DIR/$scene/$stride/$i/test/ours_30000/renders -g $MODEL_DIR/$scene/$stride/$i/test/ours_30000/gt --imgStr %05d.png --gtStr %05d.png --prefix "${scene}_${stride}"
    done
done

# DIR="/jeffyct-ssd/data/nerf_synthetic_colmap/ficus/"
# MODEL_DIR="./nerf_synthetic_colmap_test"
# for i in {0..9}; do
#     echo $i
#     python ~/Downloads/evaluate.py -i  $MODEL_DIR/$i/test/ours_30000/renders -g $MODEL_DIR/$i/test/ours_30000/gt --imgStr %05d.png --gtStr %05d.png | tee -a colmap.log
# done
