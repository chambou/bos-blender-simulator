import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter, minimum_filter
from scipy.ndimage import label
from scipy.optimize import linear_sum_assignment
from pathlib import Path
import glob, os

# ------------------------------
# CONFIGURATION
# ------------------------------

input_folder = Path("outputs/processing_results")
output_folder = Path("outputs/processing_results/stereoscopy")

def to_image(array):
    amin, amax = np.nanmin(array), np.nanmax(array)
    if amax == amin:
        return np.zeros(array.shape, dtype=np.uint8)
    norm = (array - amin) / (amax - amin)
    return (255 * norm).astype(np.uint8)

def suppress_nearby_extrema(coords, img, min_distance):
    """
    Given extrema coordinates, cluster those within min_distance and keep strongest.
    coords: [y, x] array of extrema positions
    img: the image (used to get value at each position)
    """
    if len(coords) == 0:
        return coords
    
    coords = np.array(coords)
    kept = []
    used = set()
    
    # Sort by image value (descending) so strongest are considered first
    values = img[coords[:, 0], coords[:, 1]]
    sorted_idx = np.argsort(-np.abs(values))  # Use abs for both peaks and valleys
    
    for idx in sorted_idx:
        if idx in used:
            continue
        
        y, x = coords[idx]
        kept.append((y, x))
        used.add(idx)
        
        # Mark all nearby points as used
        for jdx, (y2, x2) in enumerate(coords):
            if jdx not in used and np.hypot(y - y2, x - x2) < min_distance:
                used.add(jdx)
    
    return np.array(kept)

def remove_edge_extrema(extrema, image_shape, margin):
    """
    Remove extrema closer than `margin` pixels from any image border.
    """
    height, width = image_shape
    filtered = [
        (y, x)
        for y, x in extrema
        if margin <= y < height - margin and margin <= x < width - margin
    ]
    return np.array(filtered, dtype=int)

def compute_thresholds(img, threshold_mult):
    mean_val = np.mean(img)
    std_val = np.std(img)
    threshold_high = mean_val + threshold_mult * std_val
    threshold_low = mean_val - threshold_mult * std_val
    return threshold_high, threshold_low

def match_extrema(img_left, img_right, extrema_L, extrema_R, window_size, search_range):
    """
    Match extrema from left image to right image using local cross-correlation.
    Returns list of (y_L, x_L, y_R, x_R, disparity_x, correlation_score).
    """
    matches = []
    half_win = window_size // 2
    
    for y_L, x_L in extrema_L:
        # Extract window around extrema in left image
        y1, y2 = max(0, y_L - half_win), min(img_left.shape[0], y_L + half_win + 1)
        x1, x2 = max(0, x_L - half_win), min(img_left.shape[1], x_L + half_win + 1)
        
        if (y2 - y1) < window_size // 2 or (x2 - x1) < window_size // 2:
            continue  # Window too small near edges
        
        patch_L = img_left[y1:y2, x1:x2].astype(np.float32)
        
        # Search for best match in right image (constrained by baseline)
        # For parallel stereo, extrema should be at similar y, but different x (disparity)
        search_y_min = max(0, y_L - 10)
        search_y_max = min(img_right.shape[0], y_L + 10)
        search_x_min = max(0, x_L - search_range)
        search_x_max = min(img_right.shape[1], x_L + 1)  # Right image should be to the left or same x
        
        best_corr = -np.inf
        best_match = None
        
        for y_R in range(search_y_min, search_y_max):
            for x_R in range(search_x_min, search_x_max):
                y1_R, y2_R = max(0, y_R - half_win), min(img_right.shape[0], y_R + half_win + 1)
                x1_R, x2_R = max(0, x_R - half_win), min(img_right.shape[1], x_R + half_win + 1)
                
                if (y2_R - y1_R) < window_size // 2 or (x2_R - x1_R) < window_size // 2:
                    continue
                
                patch_R = img_right[y1_R:y2_R, x1_R:x2_R].astype(np.float32)
                
                # Normalized cross-correlation (only if patches are same size and nonzero)
                if patch_L.shape == patch_R.shape and patch_L.std() > 0 and patch_R.std() > 0:
                    corr = np.corrcoef(patch_L.ravel(), patch_R.ravel())[0, 1]
                    if corr > best_corr:
                        best_corr = corr
                        best_match = (y_R, x_R)
        
        if best_match is not None and best_corr > 0.5:  # Threshold on correlation
            y_R, x_R = best_match
            disparity = x_L - x_R  # Positive if left extrema is right of right extrema
            matches.append((y_L, x_L, y_R, x_R, disparity, best_corr))
    
    return matches

# Load phase arrays
cam0 = np.load(os.path.join(input_folder,'cam0_phase.npy'))
cam1 = np.load(os.path.join(input_folder,'cam1_phase.npy'))
img_left  = to_image(cam0)
img_right = to_image(cam1)

# Tuning parameters (adjust based on your phase pattern)
size = 101                                    # Larger neighborhood = fewer extrema
threshold_mult = 2.0                         # Stricter = fewer extrema
min_distance_NMS = 50                        # Cluster suppression radius

# Compute thresholds
mean_val = np.mean(img_left)
std_val = np.std(img_left)
threshold_high = mean_val + threshold_mult * std_val
threshold_low = mean_val - threshold_mult * std_val

# Find candidate extrema
left_threshold_high, left_threshold_low = compute_thresholds(img_left,threshold_mult)
local_max = (img_left == maximum_filter(img_left, size=size)) & (img_left > left_threshold_high)
local_min = (img_left == minimum_filter(img_left, size=size)) & (img_left < left_threshold_low)
extrema_candidates_L = np.argwhere(local_max | local_min)

# Suppress nearby ones
extrema_L = suppress_nearby_extrema(extrema_candidates_L, img_left, min_distance=min_distance_NMS)

# Same for right image
right_threshold_high, right_threshold_low = compute_thresholds(img_right,threshold_mult)
local_max_R = (img_right == maximum_filter(img_right, size=size)) & (img_right > right_threshold_high)
local_min_R = (img_right == minimum_filter(img_right, size=size)) & (img_right < right_threshold_low)
extrema_candidates_R = np.argwhere(local_max_R | local_min_R)
extrema_R = suppress_nearby_extrema(extrema_candidates_R, img_right, min_distance=min_distance_NMS)

# remove edge extrema
margin = 101 // 2
extrema_L = remove_edge_extrema(extrema_L, img_left.shape, margin)
extrema_R = remove_edge_extrema(extrema_R, img_right.shape, margin)

print(f"After filtering: {len(extrema_L)} extrema in left, {len(extrema_R)} in right")
print(extrema_L)
print(extrema_R)

# --- Match extrema across images using cross-correlation ---
window_size = 101                   # size of the local patch around extrema [px]
search_range =500                   # horizontal search range [px]
matches = match_extrema(img_left, img_right, extrema_L, extrema_R, 
                        window_size, search_range)

print(f"Matched {len(matches)} extrema pairs")
print(matches)

# --- Compute depth for each match ---
config_path = "config_stereo.json"
with open(config_path, "r") as f:
    config = json.load(f)

B = config["BOS"]["cameras_spacing"]
f_mm = config["camera"]["focal_length"]
sensor_mm = config["camera"]["sensor_size"]
W_px = config["camera"]["resolution_x"]

f_px = (f_mm / sensor_mm) * W_px

# Compute depth for each match
depths = []
for y_L, x_L, y_R, x_R, disparity, corr in matches:
    if disparity > 0.5:  # Avoid near-zero disparities
        Z = (f_px * B) / disparity
        depths.append(Z)
        print(f"Extrema at ({x_L}, {y_L}): disparity={disparity:.2f} px, depth={Z:.3f} m, corr={corr:.3f}")

# --- Estimate phase screen position ---
if depths:
    median_depth = np.median(depths)
    mean_depth = np.mean(depths)
    std_depth = np.std(depths)
    
    print(f"\n=== Phase Screen Localization ===")
    print(f"Median depth: {median_depth:.3f} m")
    print(f"Mean depth:   {mean_depth:.3f} m")
    print(f"Std dev:      {std_depth:.3f} m")
    print(f"(Depth measured from camera lens)")
else:
    print("No valid matches found.")


# Visualize matches (optional)
if depths:
    median_depth = np.median(depths)
    mean_depth = np.mean(depths)
    std_depth = np.std(depths)

    stats_text = (
        f"Median depth: {median_depth:.3f} m\n"
        f"Mean depth:   {mean_depth:.3f} m\n"
        f"Std dev:      {std_depth:.3f} m"
    )

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6))
    ax0.imshow(img_left, cmap='viridis')
    ax0.set_title('Left image with extrema')
    for y_L, x_L, y_R, x_R, disp, corr in matches[:10]:
        ax0.plot(x_L, y_L, 'r+', markersize=10)
    ax0.set_xlim(0, img_left.shape[1]); ax0.set_ylim(img_left.shape[0], 0)

    ax1.imshow(img_right, cmap='viridis')
    ax1.set_title('Right image with matched extrema')
    for y_L, x_L, y_R, x_R, disp, corr in matches[:10]:
        ax1.plot(x_R, y_R, 'r+', markersize=10)
    ax1.set_xlim(0, img_right.shape[1]); ax1.set_ylim(img_right.shape[0], 0)

    ax1.text(
        0.02, 0.98, stats_text,
        transform=ax1.transAxes,
        fontsize=10,
        color='white',
        va='top',
        ha='left',
        bbox=dict(facecolor='black', alpha=0.6, edgecolor='none')
    )

    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(output_folder,'stereo.png'), dpi=fig.dpi)