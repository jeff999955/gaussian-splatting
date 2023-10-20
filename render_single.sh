if [ $# -lt 1 ]; then
    echo "Usage: $0 <scene_id>"
    exit 1
fi

DIR=`realpath $1`
COLOR_DIR="color"
which python

mamba activate /media/jeffyct/Data/mamba/gaussian_splatting
python render.py -s $DIR \
    -m $DIR \
    -i $DIR/$COLOR_DIR \
    --data-device cpu 


