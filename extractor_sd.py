import itertools
from contextlib import ExitStack
import torch
from mask2former.data.datasets.register_ade20k_panoptic import ADE20K_150_CATEGORIES
from PIL import Image
import numpy as np
import torch.nn.functional as F
from detectron2.config import instantiate
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.config import LazyCall as L
from detectron2.data import transforms as T
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.evaluation import inference_context
from detectron2.utils.env import seed_all_rng
from detectron2.utils.visualizer import ColorMode, Visualizer, random_color
from detectron2.utils.logger import setup_logger

from odise import model_zoo
from odise.checkpoint import ODISECheckpointer
from odise.config import instantiate_odise
from odise.data import get_openseg_labels
from odise.modeling.wrapper import OpenPanopticInference

from utils.utils_correspondence import resize
import faiss


import sys


def load_model(config_path="Panoptic/odise_label_coco_50e.py", seed=42, diffusion_ver="v1-3", image_size=1024, num_timesteps=0, block_indices=(2,5,8,11), decoder_only=True, encoder_only=False, resblock_only=False):
    cfg = model_zoo.get_config(config_path, trained=True)

    cfg.model.backbone.feature_extractor.init_checkpoint = "sd://"+diffusion_ver
    cfg.model.backbone.feature_extractor.steps = (num_timesteps,)
    cfg.model.backbone.feature_extractor.unet_block_indices = block_indices
    cfg.model.backbone.feature_extractor.encoder_only = encoder_only
    cfg.model.backbone.feature_extractor.decoder_only = decoder_only
    cfg.model.backbone.feature_extractor.resblock_only = resblock_only
    cfg.model.overlap_threshold = 0
    seed_all_rng(seed)

    cfg.dataloader.test.mapper.augmentations=[
            L(T.ResizeShortestEdge)(short_edge_length=image_size, sample_style="choice", max_size=2560),
        ]
    dataset_cfg = cfg.dataloader.test

    aug = instantiate(dataset_cfg.mapper).augmentations

    model = instantiate_odise(cfg.model)
    model.to(cfg.train.device)
    ODISECheckpointer(model).load(cfg.train.init_checkpoint)

    return model, aug

def get_features(model, aug, image, caption=None, pca=False):
    height, width = image.shape[:2] if isinstance(image, np.ndarray) else (image.height, image.width)
    aug_input = T.AugInput(np.array(image), sem_seg=None)
    aug(aug_input)
    image_aug = aug_input.image
    image_tensor = torch.as_tensor(image_aug.astype("float32").transpose(2, 0, 1))  # CHW
    inputs = {"image": image_tensor, "height": height, "width": width}

    with torch.no_grad():
        if caption is not None:
            features = model.get_features([inputs], caption=caption, pca=pca)
        else:
            features = model.get_features([inputs], pca=pca)
    
    return features


def pca_process(features):
    # Get the feature tensors
    size_s5=features['s5'].shape[-1]
    size_s4=features['s4'].shape[-1]
    size_s3=features['s3'].shape[-1]

    s5 = features['s5'].reshape(features['s5'].shape[0], features['s5'].shape[1], -1)
    s4 = features['s4'].reshape(features['s4'].shape[0], features['s4'].shape[1], -1)
    s3 = features['s3'].reshape(features['s3'].shape[0], features['s3'].shape[1], -1)

    # Define the target dimensions
    target_dims = {'s5': 128, 's4': 128, 's3': 128}

    # Apply PCA to each tensor using Faiss CPU
    for name, tensor in zip(['s5', 's4', 's3'], [s5, s4, s3]):
        target_dim = target_dims[name]

        # Transpose the tensor so that the last dimension is the number of features
        tensor = tensor.permute(0, 2, 1)

        # # Norm the tensor
        # tensor = tensor / tensor.norm(dim=-1, keepdim=True)

        # Initialize a Faiss PCA object
        pca = faiss.PCAMatrix(tensor.shape[-1], target_dim)

        # Train the PCA object
        pca.train(tensor[0].cpu().numpy())

        # Apply PCA to the data
        transformed_tensor_np = pca.apply(tensor[0].cpu().numpy())

        # Convert the transformed data back to a tensor
        transformed_tensor = torch.tensor(transformed_tensor_np, device=tensor.device).unsqueeze(0)

        # Store the transformed tensor in the features dictionary
        features[name] = transformed_tensor

    # Reshape the tensors back to their original shapes
    features['s5'] = features['s5'].permute(0, 2, 1).reshape(features['s5'].shape[0], -1, size_s5, size_s5)
    features['s4'] = features['s4'].permute(0, 2, 1).reshape(features['s4'].shape[0], -1, size_s4, size_s4)
    features['s3'] = features['s3'].permute(0, 2, 1).reshape(features['s3'].shape[0], -1, size_s3, size_s3)
    # Upsample s5 spatially by a factor of 2
    upsampled_s5 = torch.nn.functional.interpolate(features['s5'], scale_factor=2, mode='bilinear', align_corners=False)

    # Concatenate upsampled_s5 and s4 to create a new s5
    features['s5'] = torch.cat((upsampled_s5, features['s4']), dim=1)

    # Set s3 as the new s4
    features['s4'] = features['s3']

    # Remove s3 from the features dictionary
    del features['s3']
    
    return features
    

def process_features_and_mask(model, aug, image, external_mask=None, input_text=None, pca=False, raw=False):

    input_image = image
    caption = input_text

    features = get_features(model, aug, input_image, caption, pca=(pca or raw))
    if pca:
        features = pca_process(features)
    if raw:
        return features
    f_s4 = features['s4']
    f_s5 = F.interpolate(features['s5'], size=f_s4.shape[-2:], mode='bilinear')
    combined = torch.cat([f_s4, f_s5], dim=1)

    if external_mask is not None:
        resized_mask = F.interpolate(external_mask.unsqueeze(0).unsqueeze(0).float(),
                                     size=f_s4.shape[-2:], mode='nearest')
        combined = combined * resized_mask
        combined[(resized_mask == 0).repeat(1, combined.shape[1], 1, 1)] = -1

    return combined


if __name__ == "__main__":
    image_path = sys.argv[1]
    try:
        input_text = sys.argv[2]
    except:
        input_text = None

    model, aug = load_model()
    img_size = 960
    image = Image.open(image_path).convert('RGB')
    image = resize(image, img_size, resize=True, to_pil=True)

    features = process_features_and_mask(model, aug, image, input_text=input_text, pca=False, raw=True)
    features = features['s4'] # save the features of layer 5
    
    # save the features
    np.save(image_path[:-4]+'.npy', features.cpu().numpy())