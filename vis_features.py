import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from extractor_sd import load_model, process_features_and_mask
from utils.utils_correspondence import co_pca, resize, find_nearest_patchs, find_nearest_patchs_replace
import matplotlib.pyplot as plt
import sys
from extractor_dino import ViTExtractor
from sklearn.decomposition import PCA as sklearnPCA
import math
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import argparse
import glob
import torchvision.transforms as transforms # 파일 상단에 추가
from PIL import ImageDraw

parser = argparse.ArgumentParser()
parser.add_argument("--model_size", type=str, default="giant")
parser.add_argument("--mask_path", type=str, default=None)
parser.add_argument("--image_path", type=str, default=None)
parser.add_argument("--output_path", type=str, default=None)
parser.add_argument("--only_dino", type=int, default=0)
parser.add_argument("--fuse_dino", type=int, default=1)
parser.add_argument("--start_idx", type=int, default=0)
args = parser.parse_args()


# hyperparameters


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
MASK = True # This now means we *use* the external mask provided, not generate one.
VER = "v1-5"
PCA = False
CO_PCA = True
PCA_DIMS = [256, 256, 256]
SIZE = 960
EDGE_PAD = False

FUSE_DINO = args.fuse_dino
ONLY_DINO = args.only_dino
DINOV2 = True
MODEL_SIZE = args.model_size


DRAW_DENSE = 1
DRAW_SWAP = 0
TEXT_INPUT = False
SEED = 42
TIMESTEP = 100

DIST = 'l2' if FUSE_DINO and not ONLY_DINO else 'cos'
if ONLY_DINO:
    FUSE_DINO = True


np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.benchmark = True

model, aug = load_model(diffusion_ver=VER, image_size=SIZE, num_timesteps=TIMESTEP)


def compute_pair_feature(model, aug, save_path, files, category, mask=False, dist='cos', real_size=960, external_mask_image1=None, external_mask_image2=None):
    if type(category) == str:
        category = [category]
    img_size = 840 if DINOV2 else 244
    model_dict={'small':'dinov2_vits14',
                'base':'dinov2_vitb14',
                'large':'dinov2_vitl14',
                'giant':'dinov2_vitg14'}

    model_type = model_dict[MODEL_SIZE] if DINOV2 else 'dino_vits8'
    layer = 11 if DINOV2 else 9
    if 'l' in model_type:
        layer = 23
    elif 'g' in model_type:
        layer = 39
    facet = 'token' if DINOV2 else 'key'
    stride = 14 if DINOV2 else 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = ViTExtractor(model_type, stride, device=device)
    patch_size = extractor.model.patch_embed.patch_size[0] if DINOV2 else extractor.model.patch_embed.patch_size
    # Calculate num_patches dynamically based on the actual output resolution of the features
    # For DINO, num_patches is typically img_size / stride
    num_patches_per_dim = img_size // stride
    num_patches = num_patches_per_dim * num_patches_per_dim


    input_text = "a photo of "+category[-1][0] if TEXT_INPUT else None

    N = len(files) // 2
    pbar = tqdm(total=N)
    result = []

    for pair_idx in range(N):

        # Load image 1
        img1 = Image.open(files[2*pair_idx]).convert('RGB')
        img1_input = resize(img1, real_size, resize=True, to_pil=True, edge=EDGE_PAD)
        img1 = resize(img1, img_size, resize=True, to_pil=True, edge=EDGE_PAD)

        # Load image 2
        img2 = Image.open(files[2*pair_idx+1]).convert('RGB')
        img2_input = resize(img2, real_size, resize=True, to_pil=True, edge=EDGE_PAD)
        img2 = resize(img2, img_size, resize=True, to_pil=True, edge=EDGE_PAD)

        # Load external masks if provided
        mask1_pil = None
        mask2_pil = None
        mask1_tensor = None
        mask2_tensor = None
        if external_mask_image1 and external_mask_image2:
            mask1_pil = Image.open(external_mask_image1).convert('L') # Load as grayscale
            mask2_pil = Image.open(external_mask_image2).convert('L')
            mask1_tensor = transforms.ToTensor()(mask1_pil).squeeze(0).float().to(device)
            mask2_tensor = transforms.ToTensor()(mask2_pil).squeeze(0).float().to(device)


        with torch.no_grad():
            # Process features for Image 1
            if not ONLY_DINO:
                img1_desc_sd = process_features_and_mask(model, aug, img1_input,
                                                        external_mask=mask1_tensor,
                                                        input_text=input_text, pca=PCA)
                # Reshape SD features to (B, num_patches, C)
                # Assuming img1_desc_sd is (B, C, H, W) where H*W is equivalent to num_patches
                b_sd, c_sd, h_sd, w_sd = img1_desc_sd.shape
                img1_desc_sd_reshaped = img1_desc_sd.permute(0, 2, 3, 1).reshape(b_sd, h_sd * w_sd, c_sd) # Should be (1, N_patches, C)
            else:
                img1_desc_sd_reshaped = None # Not used if ONLY_DINO

            if FUSE_DINO or ONLY_DINO:
                img1_batch = extractor.preprocess_pil(img1)
                img1_desc_dino = extractor.extract_descriptors(img1_batch.to(device), layer, facet).squeeze(1) # This is (1, N_patches, C)

            # Process features for Image 2
            if not ONLY_DINO:
                img2_desc_sd = process_features_and_mask(model, aug, img2_input,
                                                        external_mask=mask2_tensor,
                                                        input_text=input_text,  pca=PCA)
                b_sd, c_sd, h_sd, w_sd = img2_desc_sd.shape
                img2_desc_sd_reshaped = img2_desc_sd.permute(0, 2, 3, 1).reshape(b_sd, h_sd * w_sd, c_sd) # Should be (1, N_patches, C)
            else:
                img2_desc_sd_reshaped = None # Not used if ONLY_DINO

            if FUSE_DINO or ONLY_DINO:
                img2_batch = extractor.preprocess_pil(img2)
                img2_desc_dino = extractor.extract_descriptors(img2_batch.to(device), layer, facet).squeeze(1) # This is (1, N_patches, C)

            # Handle CO_PCA (applies to raw features, then reshaped)
            if CO_PCA:
                if not ONLY_DINO:
                    features1_raw = process_features_and_mask(model, aug, img1_input, input_text=input_text, raw=True)
                    features2_raw = process_features_and_mask(model, aug, img2_input, input_text=input_text, raw=True)
                    processed_features1_sd, processed_features2_sd = co_pca(features1_raw, features2_raw, PCA_DIMS)
                    # Reshape CO_PCA'd SD features to (B, N_patches, C)
                    b, c, h, w = processed_features1_sd.shape
                    img1_desc = processed_features1_sd.permute(0, 2, 3, 1).reshape(b, h * w, c)

                    b, c, h, w = processed_features2_sd.shape
                    img2_desc = processed_features2_sd.permute(0, 2, 3, 1).reshape(b, h * w, c)
                if FUSE_DINO: # DINO features for co-PCA are directly obtained as they are already patch-wise
                    img1_batch = extractor.preprocess_pil(img1)
                    img1_desc_dino = extractor.extract_descriptors(img1_batch.to(device), layer, facet).squeeze(1)

                    img2_batch = extractor.preprocess_pil(img2)
                    img2_desc_dino = extractor.extract_descriptors(img2_batch.to(device), layer, facet).squeeze(1)
            else: # No CO_PCA, use the already reshaped features
                img1_desc = img1_desc_sd_reshaped
                img2_desc = img2_desc_sd_reshaped

            if dist == 'l1' or dist == 'l2':
                # Normalize the features *before* concatenation if FUSE_DINO and not ONLY_DINO
                if not ONLY_DINO and img1_desc is not None:
                    img1_desc = img1_desc / img1_desc.norm(dim=-1, keepdim=True)
                    img2_desc = img2_desc / img2_desc.norm(dim=-1, keepdim=True)
                if FUSE_DINO:
                    img1_desc_dino = img1_desc_dino / img1_desc_dino.norm(dim=-1, keepdim=True)
                    img2_desc_dino = img2_desc_dino / img2_desc_dino.norm(dim=-1, keepdim=True)

            if FUSE_DINO and not ONLY_DINO:
                # Concatenate the two features together
                if img1_desc is not None:
                    # Ensure both tensors have the same number of dimensions (3D: B, N_patches, C)
                    # This is the crucial part. img1_desc_sd_reshaped and img1_desc_dino should now both be (1, N_patches, C)
                    img1_desc = torch.cat((img1_desc, img1_desc_dino), dim=-1)
                    img2_desc = torch.cat((img2_desc, img2_desc_dino), dim=-1)
                else:
                    img1_desc = img1_desc_dino
                    img2_desc = img2_desc_dino
            elif ONLY_DINO:
                img1_desc = img1_desc_dino
                img2_desc = img2_desc_dino


            # For visualization, we need the masks in the correct resolution
            # If external masks are provided, resize them to img_size for visualization
            if external_mask_image1 and external_mask_image2:
                mask1 = torch.tensor(np.array(resize(Image.open(external_mask_image1).convert('L'), img_size, resize=True, to_pil=False, edge=EDGE_PAD)) > 0).to(device)
                mask2 = torch.tensor(np.array(resize(Image.open(external_mask_image2).convert('L'), img_size, resize=True, to_pil=False, edge=EDGE_PAD)) > 0).to(device)
            else:
                # If no external mask, create a dummy full mask for visualization purposes
                mask1 = torch.ones((img_size, img_size), dtype=torch.bool).to(device)
                mask2 = torch.ones((img_size, img_size), dtype=torch.bool).to(device)

            print(mask1.shape, mask2.shape,mask1.sum(), mask2.sum())

            if DRAW_DENSE:
                if ONLY_DINO or not FUSE_DINO:
                    img1_desc = img1_desc / img1_desc.norm(dim=-1, keepdim=True)
                    img2_desc = img2_desc / img2_desc.norm(dim=-1, keepdim=True)

                # Now img1_desc and img2_desc are (B, N_patches, C)
                # Reshape for find_nearest_patchs to (B, C, H_patches, W_patches)
                img1_desc_reshaped_for_vis = img1_desc.permute(0, 2, 1).reshape(-1, img1_desc.shape[-1], num_patches_per_dim, num_patches_per_dim)
                img2_desc_reshaped_for_vis = img2_desc.permute(0, 2, 1).reshape(-1, img2_desc.shape[-1], num_patches_per_dim, num_patches_per_dim)

                trg_dense_output, src_color_map, matched_coords = find_nearest_patchs(mask2, mask1, img2, img1, img2_desc_reshaped_for_vis, img1_desc_reshaped_for_vis, mask=mask)

                if not os.path.exists(f'{save_path}'):
                    os.makedirs(f'{save_path}')
                fig_colormap, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                ax1.axis('off')
                ax2.axis('off')
                ax1.imshow(src_color_map)
                ax2.imshow(trg_dense_output)
                fig_colormap.savefig(f'{save_path}/colormap.png')
                plt.close(fig_colormap)

                max_lines_to_show = 40
                if matched_coords is not None and len(matched_coords) > max_lines_to_show:
                    import random
                    matched_coords_sampled = random.sample(matched_coords, max_lines_to_show)
                else:
                    matched_coords_sampled = matched_coords
                
                if matched_coords_sampled is not None:
                    img1_draw = img1.copy()
                    img2_draw = img2.copy()
                    draw = ImageDraw.Draw(img1_draw)
                    draw2 = ImageDraw.Draw(img2_draw)
                    canvas = Image.new('RGB', (img1.width + img2.width, max(img1.height, img2.height)))
                    canvas.paste(img1_draw, (0, 0))
                    canvas.paste(img2_draw, (img1.width, 0))
                    draw_canvas = ImageDraw.Draw(canvas)

                    for (y2, x2), (y1, x1) in matched_coords_sampled[:20]:
                        x1_px = int((x1 + 0.5) * (img1.width / num_patches_per_dim))
                        y1_px = int((y1 + 0.5) * (img1.height / num_patches_per_dim))

                        x2_px = int((x2 + 0.5) * (img2.width / num_patches_per_dim)) + img1.width
                        y2_px = int((y2 + 0.5) * (img2.height / num_patches_per_dim))

                        draw_canvas.line([(x1_px, y1_px), (x2_px, y2_px)], fill=(255, 0, 0), width=1)
                        draw_canvas.ellipse((x1_px-2, y1_px-2, x1_px+2, y1_px+2), fill=(255, 0, 0))
                        draw_canvas.ellipse((x2_px-2, y2_px-2, x2_px+2, y2_px+2), fill=(255, 0, 0))
                    
                    for (y2, x2), (y1, x1) in matched_coords_sampled[20:]:
                        x1_px = int((x1 + 0.5) * (img1.width / num_patches_per_dim))
                        y1_px = int((y1 + 0.5) * (img1.height / num_patches_per_dim))

                        x2_px = int((x2 + 0.5) * (img2.width / num_patches_per_dim)) + img1.width
                        y2_px = int((y2 + 0.5) * (img2.height / num_patches_per_dim))

                        draw_canvas.line([(x1_px, y1_px), (x2_px, y2_px)], fill=(0, 0, 255), width=1)
                        draw_canvas.ellipse((x1_px-2, y1_px-2, x1_px+2, y1_px+2), fill=(0, 0, 255))
                        draw_canvas.ellipse((x2_px-2, y2_px-2, x2_px+2, y2_px+2), fill=(0, 0, 255))
                    canvas.save(f"{save_path}/matching_lines.png")

            result.append([img1_desc.cpu(), img2_desc.cpu(), mask1.cpu(), mask2.cpu()])

    pbar.update(1)
    return result


def vis_pca_mask(result,save_path):
    # PCA visualization mask version
    for (feature1,feature2,mask1,mask2) in result:
        # feature1 shape (1,1,3600,768*2)
        # feature2 shape (1,1,3600,768*2)
        num_patches = int(math.sqrt(feature1.shape[-2]))
        # pca the concatenated feature to 3 dimensions
        feature1 = feature1.squeeze() # shape (3600,768*2)
        feature2 = feature2.squeeze() # shape (3600,768*2)
        chennel_dim = feature1.shape[-1]
        # resize back
        src_feature_reshaped = feature1.squeeze().permute(1,0).reshape(-1,num_patches,num_patches).cuda()
        tgt_feature_reshaped = feature2.squeeze().permute(1,0).reshape(-1,num_patches,num_patches).cuda()
        resized_src_mask = F.interpolate(mask1.unsqueeze(0).unsqueeze(0).float(), size=(num_patches, num_patches), mode='nearest').squeeze().cuda()
        resized_tgt_mask = F.interpolate(mask2.unsqueeze(0).unsqueeze(0).float(), size=(num_patches, num_patches), mode='nearest').squeeze().cuda()

        # Apply mask and filter out unmasked points for PCA
        src_masked_features = src_feature_reshaped * resized_src_mask.repeat(src_feature_reshaped.shape[0],1,1)
        tgt_masked_features = tgt_feature_reshaped * resized_tgt_mask.repeat(src_feature_reshaped.shape[0],1,1)

        feature1_masked_flat = src_masked_features.reshape(chennel_dim, -1).permute(1,0)
        feature2_masked_flat = tgt_masked_features.reshape(chennel_dim, -1).permute(1,0)

        # Filter out rows where all elements are zero (due to mask)
        feature1_valid_indices = torch.where(feature1_masked_flat.sum(dim=1) != 0)[0]
        feature2_valid_indices = torch.where(feature2_masked_flat.sum(dim=1) != 0)[0]

        feature1_filtered = feature1_masked_flat[feature1_valid_indices]
        feature2_filtered = feature2_masked_flat[feature2_valid_indices]

        #####################token개수################################33
        print(f"Masked DINO+SD tokens for image 1: {feature1_filtered.shape[0]} (out of {feature1_masked_flat.shape[0]})")
        print(f"Masked DINO+SD tokens for image 2: {feature2_filtered.shape[0]} (out of {feature2_masked_flat.shape[0]})")
        ##########################################################


        n_components=4 # the first component is to seperate the object from the background
        pca = sklearnPCA(n_components=n_components)

        # Ensure there are enough samples for PCA
        if feature1_filtered.shape[0] == 0 or feature2_filtered.shape[0] == 0:
            print("Warning: One or both masks are empty after resizing. Skipping PCA visualization.")
            continue

        feature1_n_feature2_filtered = torch.cat((feature1_filtered, feature2_filtered), dim=0)
        feature1_n_feature2_filtered = pca.fit_transform(feature1_n_feature2_filtered.cpu().numpy())

        # Map PCA results back to original patch locations
        feature1_pca_full = np.zeros((num_patches**2, n_components))
        feature1_pca_full[feature1_valid_indices.cpu().numpy()] = feature1_n_feature2_filtered[:feature1_filtered.shape[0],:]

        feature2_pca_full = np.zeros((num_patches**2, n_components))
        feature2_pca_full[feature2_valid_indices.cpu().numpy()] = feature1_n_feature2_filtered[feature1_filtered.shape[0]:,:]

        fig, axes = plt.subplots(4, 2, figsize=(10, 14))
        for show_channel in range(n_components):
            if show_channel==0:
                continue
            # min max normalize the feature map for display
            feature1_channel_display = feature1_pca_full[:, show_channel]
            feature2_channel_display = feature2_pca_full[:, show_channel]

            # Only normalize non-zero values for display
            if feature1_channel_display.max() - feature1_channel_display.min() > 0:
                feature1_channel_display = (feature1_channel_display - feature1_channel_display.min()) / (feature1_channel_display.max() - feature1_channel_display.min())
            if feature2_channel_display.max() - feature2_channel_display.min() > 0:
                feature2_channel_display = (feature2_channel_display - feature2_channel_display.min()) / (feature2_channel_display.max() - feature2_channel_display.min())

            feature1_first_channel = feature1_channel_display.reshape(num_patches,num_patches)
            feature2_first_channel = feature2_channel_display.reshape(num_patches,num_patches)

            axes[show_channel-1, 0].imshow(feature1_first_channel * resized_src_mask.cpu().numpy()) # Apply mask to visualization
            axes[show_channel-1, 0].axis('off')
            axes[show_channel-1, 1].imshow(feature2_first_channel * resized_tgt_mask.cpu().numpy()) # Apply mask to visualization
            axes[show_channel-1, 1].axis('off')
            axes[show_channel-1, 0].set_title('Feature 1 - Channel {}'.format(show_channel ), fontsize=14)
            axes[show_channel-1, 1].set_title('Feature 2 - Channel {}'.format(show_channel ), fontsize=14)


        feature1_resized = feature1_pca_full[:, 1:4].reshape(num_patches,num_patches, 3)
        feature2_resized = feature2_pca_full[:, 1:4].reshape(num_patches,num_patches, 3)

        axes[3, 0].imshow(feature1_resized * np.expand_dims(resized_src_mask.cpu().numpy(), axis=-1))
        axes[3, 0].axis('off')
        axes[3, 1].imshow(feature2_resized * np.expand_dims(resized_tgt_mask.cpu().numpy(), axis=-1))
        axes[3, 1].axis('off')
        axes[3, 0].set_title('Feature 1 - All Channels', fontsize=14)
        axes[3, 1].set_title('Feature 2 - All Channels', fontsize=14)

        plt.tight_layout()
        plt.show()
        fig.savefig(save_path+'/masked_pca.png', dpi=300)

def vis_pca(result, save_path, src_img_path, trg_img_path):
    # PCA visualization (without explicit mask application during PCA, but applied during display)
    for (feature1, feature2, mask1, mask2) in result: # mask1 and mask2 are now the resized (to img_size) ground truth masks
        # Dynamically calculate num_patches_per_dim based on the expected resolution of the features
        # and the stride of the DINO model. This ensures consistency with feature extraction.
        img_size_for_dino = 840 if DINOV2 else 244 # This should match the img_size used in compute_pair_feature for DINO preprocessing
        stride_for_dino = 14 if DINOV2 else 4 # This should match the stride used in ViTExtractor
        num_patches_per_dim = img_size_for_dino // stride_for_dino
        
        # Squeeze the batch dimension (1) so feature1 and feature2 are (N_patches_total, C)
        feature1 = feature1.squeeze(0) # shape (N_patches_total, C)
        feature2 = feature2.squeeze(0) # shape (N_patches_total, C)
        chennel_dim = feature1.shape[-1]

        # Reshape to a 3D tensor: (height_of_patches, width_of_patches, chennel_dim)
        # num_patches_per_dim is now explicitly used for the spatial dimensions
        feature1_reshaped = feature1.reshape(num_patches_per_dim, num_patches_per_dim, chennel_dim)
        feature2_reshaped = feature2.reshape(num_patches_per_dim, num_patches_per_dim, chennel_dim)

        h1_orig, w1_orig = Image.open(src_img_path).size
        h2_orig, w2_orig = Image.open(trg_img_path).size

        # Recalculate aspect ratio for cropping/padding to match num_patches square grid
        # Use SIZE for input image scaling, but apply patch-level logic for feature maps
        img_size_for_pca_crop = SIZE # Original image size used for input to model, e.g., 960

        # Calculate dimensions for feature map cropping based on original image aspect ratio
        # and how the features were extracted (stride)
        
        # --- Feature 1 (Source Image) ---
        if EDGE_PAD:
            # If edge padding, features are already num_patches_per_dim x num_patches_per_dim
            feature1_uncropped = feature1_reshaped
        else:
            # Determine actual feature map dimensions based on the original image aspect ratio and scaling
            # This logic needs to align with how resize and feature extraction were done in compute_pair_feature
            
            # First, determine the effective input image size after resize (similar to how img1_input was generated)
            if h1_orig > w1_orig:
                # Scaled to SIZE height, width proportionally
                effective_h1_input = img_size_for_pca_crop
                effective_w1_input = int(w1_orig * (img_size_for_pca_crop / h1_orig))
            else:
                # Scaled to SIZE width, height proportionally
                effective_w1_input = img_size_for_pca_crop
                effective_h1_input = int(h1_orig * (img_size_for_pca_crop / w1_orig))
            
            # Then, calculate the patch dimensions (H_patches, W_patches)
            # This is critical: only these central patches contain valid information if not EDGE_PAD
            patch_h1_valid = int(effective_h1_input / stride_for_dino)
            patch_w1_valid = int(effective_w1_input / stride_for_dino)

            # Calculate start and end indices for cropping from the num_patches_per_dim square grid
            start_h1 = (num_patches_per_dim - patch_h1_valid) // 2
            start_w1 = (num_patches_per_dim - patch_w1_valid) // 2
            
            feature1_uncropped = feature1_reshaped[
                start_h1 : start_h1 + patch_h1_valid,
                start_w1 : start_w1 + patch_w1_valid,
                :
            ]

        f1_shape = feature1_uncropped.shape[:2] # (H_patches, W_patches) of valid region
        feature1_flat = feature1_uncropped.reshape(f1_shape[0] * f1_shape[1], chennel_dim)

        # --- Feature 2 (Target Image) ---
        if EDGE_PAD:
            feature2_uncropped = feature2_reshaped
        else:
            if h2_orig > w2_orig:
                effective_h2_input = img_size_for_pca_crop
                effective_w2_input = int(w2_orig * (img_size_for_pca_crop / h2_orig))
            else:
                effective_w2_input = img_size_for_pca_crop
                effective_h2_input = int(h2_orig * (img_size_for_pca_crop / w2_orig))

            patch_h2_valid = int(effective_h2_input / stride_for_dino)
            patch_w2_valid = int(effective_w2_input / stride_for_dino)

            start_h2 = (num_patches_per_dim - patch_h2_valid) // 2
            start_w2 = (num_patches_per_dim - patch_w2_valid) // 2

            feature2_uncropped = feature2_reshaped[
                start_h2 : start_h2 + patch_h2_valid,
                start_w2 : start_w2 + patch_w2_valid,
                :
            ]

        f2_shape = feature2_uncropped.shape[:2] # (H_patches, W_patches) of valid region
        feature2_flat = feature2_uncropped.reshape(f2_shape[0] * f2_shape[1], chennel_dim)

        n_components = 3
        pca = sklearnPCA(n_components=n_components)
        feature1_n_feature2 = torch.cat((feature1_flat, feature2_flat), dim=0)
        feature1_n_feature2 = pca.fit_transform(feature1_n_feature2.cpu().numpy())
        feature1_pca = feature1_n_feature2[:feature1_flat.shape[0], :]
        feature2_pca = feature1_n_feature2[feature1_flat.shape[0]:, :]

        # Resize masks to feature map resolution for visualization
        # mask1 and mask2 are now the resized (to img_size) ground truth masks from compute_pair_feature
        # We need to interpolate them to the *actual* (cropped) feature map dimensions (f1_shape, f2_shape)
        resized_src_mask_vis = F.interpolate(mask1.unsqueeze(0).unsqueeze(0).float(), size=f1_shape, mode='nearest').squeeze().cpu().numpy()
        resized_tgt_mask_vis = F.interpolate(mask2.unsqueeze(0).unsqueeze(0).float(), size=f2_shape, mode='nearest').squeeze().cpu().numpy()

        fig, axes = plt.subplots(4, 2, figsize=(10, 14))
        for show_channel in range(n_components):
            # min max normalize the feature map
            feature1_channel_display = (feature1_pca[:, show_channel] - feature1_pca[:, show_channel].min()) / (feature1_pca[:, show_channel].max() - feature1_pca[:, show_channel].min())
            feature2_channel_display = (feature2_pca[:, show_channel] - feature2_pca[:, show_channel].min()) / (feature2_pca[:, show_channel].max() - feature2_pca[:, show_channel].min())

            feature1_first_channel = feature1_channel_display.reshape(f1_shape[0], f1_shape[1])
            feature2_first_channel = feature2_channel_display.reshape(f2_shape[0], f2_shape[1])

            axes[show_channel, 0].imshow(feature1_first_channel) 
            axes[show_channel, 0].axis('off')
            axes[show_channel, 1].imshow(feature2_first_channel) 
            axes[show_channel, 1].axis('off')
            axes[show_channel, 0].set_title('Feature 1 - Channel {}'.format(show_channel + 1), fontsize=14)
            axes[show_channel, 1].set_title('Feature 2 - Channel {}'.format(show_channel + 1), fontsize=14)



        feature1_resized = feature1_pca[:, :3].reshape(f1_shape[0], f1_shape[1], 3)
        feature2_resized = feature2_pca[:, :3].reshape(f2_shape[0], f2_shape[1], 3)

        axes[3, 0].imshow(feature1_resized * np.expand_dims(resized_src_mask_vis, axis=-1))
        axes[3, 0].axis('off')
        axes[3, 1].imshow(feature2_resized * np.expand_dims(resized_tgt_mask_vis, axis=-1))
        axes[3, 1].axis('off')
        axes[3, 0].set_title('Feature 1 - All Channels', fontsize=14)
        axes[3, 1].set_title('Feature 2 - All Channels', fontsize=14)

        plt.tight_layout()
        plt.show()
        fig.savefig(os.path.join(save_path, 'pca.png'), dpi=300)


def perform_clustering(features, n_clusters=10):
    # Normalize features
    features = F.normalize(features, p=2, dim=1)
    # Convert the features to float32
    features = features.cpu().detach().numpy().astype('float32')
    # Initialize a k-means clustering index with the desired number of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto') # Added n_init for KMeans
    # Train the k-means index with the features
    kmeans.fit(features)
    # Assign the features to their nearest cluster
    labels = kmeans.predict(features)

    return labels

def cluster_and_match(result, save_path, n_clusters=6):
    for (feature1,feature2,mask1,mask2) in result:
        num_patches = int(math.sqrt(feature1.shape[-2]))
        feature1 = feature1.squeeze()
        feature2 = feature2.squeeze()
        chennel_dim = feature1.shape[-1]

        src_feature_reshaped = feature1.squeeze().permute(1,0).reshape(-1,num_patches,num_patches).cuda()
        tgt_feature_reshaped = feature2.squeeze().permute(1,0).reshape(-1,num_patches,num_patches).cuda()

        # Resize masks to feature map resolution
        resized_src_mask = F.interpolate(mask1.unsqueeze(0).unsqueeze(0).float(), size=(num_patches, num_patches), mode='nearest').squeeze().cuda()
        resized_tgt_mask = F.interpolate(mask2.unsqueeze(0).unsqueeze(0).float(), size=(num_patches, num_patches), mode='nearest').squeeze().cuda()

        # Apply mask and filter out unmasked points for clustering
        src_masked_features = src_feature_reshaped * resized_src_mask.repeat(src_feature_reshaped.shape[0],1,1)
        tgt_masked_features = tgt_feature_reshaped * resized_tgt_mask.repeat(src_feature_reshaped.shape[0],1,1)

        features1_2d = src_masked_features.reshape(chennel_dim, -1).permute(1,0)
        features2_2d = tgt_masked_features.reshape(chennel_dim, -1).permute(1,0)

        # Filter out rows where all elements are zero (due to mask)
        feature1_valid_indices = torch.where(features1_2d.sum(dim=1) != 0)[0]
        feature2_valid_indices = torch.where(features2_2d.sum(dim=1) != 0)[0]

        features1_filtered = features1_2d[feature1_valid_indices]
        features2_filtered = features2_2d[feature2_valid_indices]

        if features1_filtered.shape[0] == 0 or features2_filtered.shape[0] == 0:
            print("Warning: One or both masks are empty after resizing. Skipping clustering.")
            continue

        labels_img1_filtered = perform_clustering(features1_filtered, n_clusters)
        labels_img2_filtered = perform_clustering(features2_filtered, n_clusters)

        # Create full label maps and fill with -1 for masked out areas
        labels_img1_full = np.full(num_patches * num_patches, -1, dtype=int)
        labels_img1_full[feature1_valid_indices.cpu().numpy()] = labels_img1_filtered

        labels_img2_full = np.full(num_patches * num_patches, -1, dtype=int)
        labels_img2_full[feature2_valid_indices.cpu().numpy()] = labels_img2_filtered


        # Calculate cluster means only for valid points
        cluster_means_img1 = []
        for i in range(n_clusters):
            cluster_points = features1_filtered.cpu().detach().numpy()[labels_img1_filtered == i]
            if len(cluster_points) > 0:
                cluster_means_img1.append(cluster_points.mean(axis=0))
            else:
                cluster_means_img1.append(np.zeros(chennel_dim)) # Handle empty clusters
        cluster_means_img1 = np.array(cluster_means_img1)


        cluster_means_img2 = []
        for i in range(n_clusters):
            cluster_points = features2_filtered.cpu().detach().numpy()[labels_img2_filtered == i]
            if len(cluster_points) > 0:
                cluster_means_img2.append(cluster_points.mean(axis=0))
            else:
                cluster_means_img2.append(np.zeros(chennel_dim))
        cluster_means_img2 = np.array(cluster_means_img2)

        distances = np.linalg.norm(np.expand_dims(cluster_means_img1, axis=1) - np.expand_dims(cluster_means_img2, axis=0), axis=-1)
        row_ind, col_ind = linear_sum_assignment(distances)

        relabeled_img2_full = np.copy(labels_img2_full)
        for i, match in zip(row_ind, col_ind):
            # Only relabel if the cluster was not masked out initially
            relabeled_img2_full[labels_img2_full == match] = i

        labels_img1_display = labels_img1_full.reshape(num_patches, num_patches)
        relabeled_img2_display = relabeled_img2_full.reshape(num_patches, num_patches)

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        ax_img1 = axs[0]
        axs[0].axis('off')
        ax_img1.imshow(labels_img1_display, cmap='tab20')

        ax_img2 = axs[1]
        axs[1].axis('off')
        ax_img2.imshow(relabeled_img2_display, cmap='tab20')

        plt.tight_layout()
        plt.show()
        fig.savefig(save_path+'/clustering.png', dpi=300)



def process_images(src_img_path, trg_img_path, src_mask_path, trg_mask_path):

    categories = [['garment'], ['upper_clothes']]
    files = [src_img_path, trg_img_path]


    save_folder = args.output_path
    img_id = os.path.basename(os.path.dirname(trg_img_path))
    model_id = os.path.basename(trg_img_path).split('.')[0]
    garment_id = os.path.basename(src_img_path).split('.')[0]
    save_path = f"{save_folder}/{img_id}/{garment_id}_{model_id}"
    os.makedirs(save_path, exist_ok=True)

    result = compute_pair_feature(model, aug, save_path, files, mask=MASK, category=categories, dist=DIST,
                                external_mask_image1=src_mask_path, external_mask_image2=trg_mask_path)

    if MASK:
        vis_pca_mask(result, save_path)
        cluster_and_match(result, save_path)
    
    ##################mask 없이 pca###################
    # vis_pca(result, save_path, src_img_path, trg_img_path)

    return result

# Main execution loop
target_img_list = glob.glob(f"{args.image_path}/*/model_*.jpg")
target_img_list = sorted(target_img_list)
# breakpoint()
target_img_list.remove("/home/elicer/projects/dahyun/20251R0136COSE40500/dataset/images/4849/model_0.jpg")
target_img_list.remove("/home/elicer/projects/dahyun/20251R0136COSE40500/dataset/images/4849/model_1.jpg")
target_img_list.remove("/home/elicer/projects/dahyun/20251R0136COSE40500/dataset/images/4849/model_2.jpg")

for idx, tgt_img in enumerate(target_img_list):
    if idx % 2 == 0:
        img_id = os.path.basename(os.path.dirname(tgt_img))
        
        src_img = f"{os.path.dirname(tgt_img)}/product_front.png"
        src_mask = f"{args.mask_path}/{img_id}/product_front.png"  
        tgt_mask = f"{args.mask_path}/{img_id}/model_upper_mask_00.png" 
        result = process_images(src_img, tgt_img, src_mask, tgt_mask)
    elif idx % 2 == 1:
        src_img = f"{os.path.dirname(tgt_img)}/product_back.png"
        src_mask = f"{args.mask_path}/{img_id}/product_back.png" 
        tgt_mask = f"{args.mask_path}/{img_id}/model_upper_mask_01.png" 
        result = process_images(src_img, tgt_img, src_mask, tgt_mask)

