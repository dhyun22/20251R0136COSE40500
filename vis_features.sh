export CUDA_VISIBLE_DEVICES=1
python vis_features.py \
    --model_size giant \
    --mask_path /home/elicer/projects/dahyun/sd-dino/dataset/masks \
    --image_path /home/elicer/projects/dahyun/sd-dino/dataset/images \
    --output_path /home/elicer/projects/dahyun/sd-dino/dataset/output \
    --only_dino 1 \
    --fuse_dino 0 \
    --start_idx 0