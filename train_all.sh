
COLOR_DIR="color"

mamba activate /media/jeffyct/Data/mamba/gaussian_splatting
which python
sc=(scene0101_04 scene0241_01)

for s in ${sc[@]}; do
    DIR=/media/jeffyct/Data/ScanNet/scans/$s
    python train.py -s $DIR \
        -m $DIR \
        -i $DIR/$COLOR_DIR \
        --data-device cpu \
        --eval
done

for s in ${sc[@]}; do
    DIR=/media/jeffyct/Data/ScanNet/scans/$s
    python render.py -s $DIR \
        -m $DIR \
        -i $DIR/$COLOR_DIR \
        --data-device cpu \
        --eval 
done
