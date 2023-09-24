CUR=$(pwd)
conda activate gaussian_splatting
for d in /scratch/data/Replica/*; do
    if [ -d $d ]; then
        echo $d
        zip -ur replica.zip $d/test/ours_30000/renders/*.png
    fi
done
