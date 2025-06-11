export CUDA_VISIBLE_DEVICES=0
python vis_features_model.py \
    --model_size giant \
    --mask_path /home/elicer/projects/dahyun/20251R0136COSE40500/dataset/masks \
    --image_path /home/elicer/projects/dahyun/20251R0136COSE40500/dataset/images \
    --output_path /home/elicer/projects/dahyun/20251R0136COSE40500/dataset/output_model \
    --only_dino 0 \
    --fuse_dino 1 \
    --start_idx 0