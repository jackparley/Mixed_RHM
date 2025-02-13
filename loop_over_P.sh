for pp in 128 256 384 640 1024 1792 3200 5376 9216 15616
do
    sbatch v_16_L_2.sh "$pp"
done