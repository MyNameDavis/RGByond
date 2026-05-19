"""
================================================================================
                        THE RGByond the Pale Toolkit
================================================================================

PRIMER & CORE CONCEPTS:
Digital images are typically processed in RGB space, but human color perception 
aligns far better with HSV (Hue, Saturation, Value). This toolkit provides a 
suite of mathematical re-mappings within the HSV color space to alter, intensify, 
and maximize color diversity across entire images or localized spatial regions.

Key Strategies Included:
1. Global Range Stretching (Percentile / Interpolation): Ignores outlier pixels 
   to forcefully stretch the core color distribution across the full spectrum.
2. Histogram Equalization: Maps dominant, overrepresented colors into 
   underrepresented hue spaces based on cumulative distribution functions (CDF).
3. Contrast Limited Adaptive Histogram Equalization (CLAHE): Breaks down images 
   into local tiles, mapping color and brightness changes locally to solve spatial 
   uniformity issues without generating high-frequency digital noise.
4. Channel Swapping & Bending: Non-linear modifications altering structural 
   relationships between brightness and color matrices.
"""

import os
import cv2
import rawpy
import numpy as np
from PIL import Image
import matplotlib.colors as mcolors
from scipy.interpolate import interp1d
from skimage import exposure
from pillow_heif import register_heif_opener

# Enable HEIC/HEIF processing natively in Pillow
register_heif_opener()

# ================================================================================
# 1. IMAGE LOADING PIPELINE
# ================================================================================

def load_image(file_path):
    """
    Automatically detects image format by extension and processes it safely into
    a standard PIL Image object. Handles RAW (.ARW), HEIC, and traditional standard formats.
    
    Parameters:
        file_path (str): Relative or absolute path to the input image.
        
    Returns:
        PIL.Image: Python Imaging Library image object normalized to standard RGB mode.
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    print(f"[Loader] Initializing processing for format: '{ext}'")
    
    try:
        if ext in ['.arw', '.cr2', '.nef', '.dng']:
            # Handle RAW camera image data
            with rawpy.imread(file_path) as raw:
                rgb_array = raw.postprocess()
            return Image.fromarray(rgb_array)
            
        elif ext in ['.heic', '.heif', '.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
            # Handle HEIC (via registered opener) and traditional standard formats
            img = Image.open(file_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return img
            
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
            
    except Exception as e:
        print(f"[Error] Failed to load file at {file_path}. Details: {e}")
        raise

# ================================================================================
# 2. TRANSFORMATION TOOLKIT UTILITIES
# ================================================================================

def HV_swap(img, power_law=[1.0, 1.0, 1.0]):
    """
    Exchanges the Hue matrix with the Value matrix, cross-wiring perceived 
    color with perceived luminosity structure.
    
    Parameters:
        img (PIL.Image): Input image.
        power_law (list of float): Nonlinear gamma scaling factors [V_gamma, S_gamma, H_gamma].
                                  Values < 1.0 expand dark values, values > 1.0 compress them.
    Returns:
        PIL.Image: Swapped and modified image.
    """
    img_rgb_array = np.array(img)
    img_hsv_array = mcolors.rgb_to_hsv(img_rgb_array / 255.0)
    H, S, V = img_hsv_array[:,:,0], img_hsv_array[:,:,1], img_hsv_array[:,:,2]
    
    # Cross-wire channels: Old Value becomes Hue, Old Hue becomes Value
    img_hsv_array_swap = np.stack([
        np.clip(V ** power_law[0], 0.0, 1.0),
        np.clip(S ** power_law[1], 0.0, 1.0),
        np.clip(H ** power_law[2], 0.0, 1.0)
    ], axis=2)
    
    img_rgb_array_swap = np.uint8(mcolors.hsv_to_rgb(img_hsv_array_swap) * 255)
    return Image.fromarray(img_rgb_array_swap)


def HSV_bend(img, power_law=[1.0, 1.0, 1.0]):
    """
    Applies independent non-linear gamma curves (power-law functions) across 
    all three channels simultaneously.
    
    Parameters:
        img (PIL.Image): Input image.
        power_law (list of float): Curves for [H_gamma, S_gamma, V_gamma].
                                  Tweak S_gamma down (e.g., 0.5) to aggressively pop neutral colors.
    Returns:
        PIL.Image: Tone and color bent image.
    """
    img_rgb_array = np.array(img)
    img_hsv_array = mcolors.rgb_to_hsv(img_rgb_array / 255.0)
    H, S, V = img_hsv_array[:,:,0], img_hsv_array[:,:,1], img_hsv_array[:,:,2]
    
    img_hsv_array_swap = np.stack([
        np.clip(H ** power_law[0], 0.0, 1.0),
        np.clip(S ** power_law[1], 0.0, 1.0),
        np.clip(V ** power_law[2], 0.0, 1.0)
    ], axis=2)
    
    img_rgb_array_swap = np.uint8(mcolors.hsv_to_rgb(img_hsv_array_swap) * 255)
    return Image.fromarray(img_rgb_array_swap)


def hue_spread(img, kind='linear', saturation_boost=1.3):
    """
    Calculates statistical bounds of color distribution discarding outliers, 
    and linearly stretches them across the full [0.0, 1.0] color spectrum.
    
    Parameters:
        img (PIL.Image): Input image.
        kind (str): Interpolation type ('linear', 'quadratic', 'cubic').
        saturation_boost (float): Multiplier for color depth to clear out muddy hues.
    Returns:
        PIL.Image: Spectrum optimized image.
    """
    img_rgb_array = np.array(img)
    img_hsv_array = mcolors.rgb_to_hsv(img_rgb_array / 255.0)
    H, S, V = img_hsv_array[:,:,0], img_hsv_array[:,:,1], img_hsv_array[:,:,2]
    
    # Utilizing high/low percentiles cuts out dust/noise tracking faults
    H_range_actual = [np.percentile(H, 0.01), np.percentile(H, 99.99)]
    H_range_full = [0.0, 1.0]
    
    map_func = interp1d(H_range_actual, H_range_full, kind=kind, fill_value='extrapolate')
    H_stretched = np.clip(map_func(H), 0.0, 1.0)
    
    img_HSV_array_spread = np.stack([
        H_stretched, 
        np.clip(S * saturation_boost, 0.0, 1.0), 
        V
    ], axis=2)
    
    img_rgb_array_spread = np.uint8(mcolors.hsv_to_rgb(img_HSV_array_spread) * 255)
    return Image.fromarray(img_rgb_array_spread)


def extreme_rainbow_spread(img, saturation_boost=1.3):
    """
    Performs global Histogram Equalization explicitly on Hue. Dominant tones 
    are flattened out across their spatial neighbors, flooding empty ranges.
    
    Parameters:
        img (PIL.Image): Input image.
        saturation_boost (float): Color amplification scale anchor.
    Returns:
        PIL.Image: Equalized, maximum variety output image.
    """
    img_rgb_array = np.array(img)
    img_hsv_array = mcolors.rgb_to_hsv(img_rgb_array / 255.0)
    H, S, V = img_hsv_array[:,:,0], img_hsv_array[:,:,1], img_hsv_array[:,:,2]

    H_flat = H.flatten()
    hist, bins = np.histogram(H_flat, bins=256, range=(0.0, 1.0))
    cdf = hist.cumsum()

    # Re-normalize CDF to fit strictly within 0-1 boundaries
    cdf_normalized = (cdf - cdf.min()) / (cdf.max() - cdf.min())
    H_spread = np.interp(H_flat, bins[:-1], cdf_normalized).reshape(H.shape)

    img_HSV_spread = np.stack([
        H_spread, 
        np.clip(S * saturation_boost, 0.0, 1.0), 
        V
    ], axis=2)
    
    img_rgb_spread = np.uint8(mcolors.hsv_to_rgb(img_HSV_spread) * 255)
    return Image.fromarray(img_rgb_spread)


def hue_spatial_clahe_opencv(img, clip_limit=3.0, tile_grid=(40, 40), saturation_factor=1.1, value_factor=0.9):
    """
    Executes Adaptive Histogram Equalization locally on the Hue spectrum. 
    Forces monochromatic zones to yield a vibrant rainbow layout matching spatial boundaries.
    
    Parameters:
        img (PIL.Image): Input image.
        clip_limit (float): Higher limits (e.g. 4.0) force maximum divergence. Lower limits hold realism.
        tile_grid (tuple): Layout of local scanning blocks. (40,40) handles ultra-fine variations.
        saturation_factor (float): Multiplier for color vividness.
        value_factor (float): Dynamic tuning factor to brighten or darken global results.
    Returns:
        PIL.Image: Spatially adaptive balanced output image.
    """
    img_rgb_array = np.array(img)
    img_hsv_array = mcolors.rgb_to_hsv(img_rgb_array / 255.0)
    H, S, V = img_hsv_array[:,:,0], img_hsv_array[:,:,1], img_hsv_array[:,:,2]

    H_uint8 = np.uint8(H * 255)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    H_adaptive_uint8 = clahe.apply(H_uint8)
    H_adaptive = H_adaptive_uint8 / 255.0

    img_HSV_adaptive = np.stack([
        H_adaptive, 
        np.clip(S * saturation_factor, 0.0, 1.0), 
        np.clip(V * value_factor, 0.0, 1.0)
    ], axis=2)
    
    img_rgb_adaptive = np.uint8(mcolors.hsv_to_rgb(img_HSV_adaptive) * 255)
    return Image.fromarray(img_rgb_adaptive)


def hue_spatial_clahe_opencv_VALUE(img, clip_limit=1.0, tile_grid=(10, 10), saturation_factor=1.0):
    """
    Applies spatial CLAHE localized map adjustment only to the Value (brightness) channel.
    Unveils massive structural detail across shadowed matrices while holding Hue constant.
    
    Parameters:
        img (PIL.Image): Input image.
        clip_limit (float): Threshold determining local structural pop (1.0 to 3.0 typical).
        tile_grid (tuple): Row/Col subdivision grid. Smaller grid squares yield localized contrast.
        saturation_factor (float): Adjusts output intensity.
    Returns:
        PIL.Image: Multi-scale micro-contrast mapped image.
    """
    img_rgb_array = np.array(img)
    img_hsv_array = mcolors.rgb_to_hsv(img_rgb_array / 255.0)
    H, S, V = img_hsv_array[:,:,0], img_hsv_array[:,:,1], img_hsv_array[:,:,2]

    V_uint8 = np.uint8(V * 255)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    V_adaptive_uint8 = clahe.apply(V_uint8)
    V_adaptive = V_adaptive_uint8 / 255.0

    img_HSV_adaptive = np.stack([
        H, 
        np.clip(S * saturation_factor, 0.0, 1.0), 
        V_adaptive
    ], axis=2)
    
    img_rgb_adaptive = np.uint8(mcolors.hsv_to_rgb(img_HSV_adaptive) * 255)
    return Image.fromarray(img_rgb_adaptive)


def dual_channel_clahe(img, hue_limit=0.01, value_limit=0.03, saturation_boost=1.1):
    """
    Simultaneously applies adaptive histogram equalization across both Hue and Value channels.
    Locks extreme details and microtonal variation without destroying photo geometry.
    
    Parameters:
        img (PIL.Image): Input image.
        hue_limit (float): Controls color mutation thresholds. Keep low (0.005 - 0.02) to maintain realism.
        value_limit (float): Controls tone mapping. (0.02 - 0.05) extracts dramatic textures.
        saturation_boost (float): Linear enhancement tracking newly discovered gradients.
    Returns:
        PIL.Image: Highly detailed, dynamically color balanced image.
    """
    img_rgb_array = np.array(img)
    img_hsv_array = mcolors.rgb_to_hsv(img_rgb_array / 255.0)
    H, S, V = img_hsv_array[:,:,0], img_hsv_array[:,:,1], img_hsv_array[:,:,2]

    H_adaptive = exposure.equalize_adapthist(H, kernel_size=None, clip_limit=hue_limit)
    V_adaptive = exposure.equalize_adapthist(V, kernel_size=None, clip_limit=value_limit)

    img_HSV_dual = np.stack([
        H_adaptive, 
        np.clip(S * saturation_boost, 0.0, 1.0), 
        V_adaptive
    ], axis=2)
    
    img_rgb_dual = np.uint8(mcolors.hsv_to_rgb(img_HSV_dual) * 255)
    return Image.fromarray(img_rgb_dual)

# ================================================================================
# 3. CONTROL CENTER EXECUTION BLOCK
# ================================================================================

if __name__ == "__main__":
    # Define file paths
    INPUT_PATH = "[IMAGE INPUT]"
    OUTPUT_PATH = "[IMAGE OUTPUT]"
    
    # 1. Load the image via the smart pipeline
    loaded_img = load_image(INPUT_PATH)
    
    # 2. CHOOSE YOUR TOOL
    # To switch between transformations, simply uncomment the line you want to run:
    
    processed_img = HV_swap(loaded_img, power_law=[1.0, 1.0, 0.5])
    # processed_img = HSV_bend(loaded_img, power_law=[1.0, 0.166, 1.0])
    # processed_img = hue_spread(loaded_img, kind='linear', saturation_boost=1.3)
    # processed_img = extreme_rainbow_spread(loaded_img, saturation_boost=1.3)
    # processed_img = hue_spatial_clahe_opencv(loaded_img, clip_limit=3.0, tile_grid=(40,40))
    # processed_img = hue_spatial_clahe_opencv_VALUE(loaded_img, clip_limit=1.0, tile_grid=(10,10))
    # processed_img = dual_channel_clahe(loaded_img, hue_limit=0.01, value_limit=0.03, saturation_boost=1.1)

    # 3. Save the finalized asset
    processed_img.save(OUTPUT_PATH, quality=95)
    print(f"[Success] Saved transformed output to target: '{OUTPUT_PATH}'")