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

def compute_thresholds(img, threshold_mult):
    # This computes a threshold to what counts as extrema, if theyre higher than the high and lower than the low value.
    # defined by the the values of the mean plus/minus a factor of its standard deviation
    mean_val = np.mean(img)
    std_val = np.std(img)
    threshold_high = mean_val + threshold_mult * std_val
    threshold_low = mean_val - threshold_mult * std_val
    return threshold_high, threshold_low

def match_extrema(img_left, img_right, extrema_L, window_size, search_range):
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

def correlation_1d(img_left, img_right, kernel_center, kernel_width):
    correlation = []
    kernel_height = img_left.shape[0]
    half_win_width = kernel_width // 2
    img_center = kernel_center

    # Extract window
    y1, y2 = 0, img_left.shape[0]
    x1 = img_center - half_win_width
    x2 = x1 + kernel_width

    best_corr = -np.inf
    best_x_R = None

    # create left image patch
    patch_L = img_left[y1:y2, x1:x2].astype(np.float32)

    # # Search for best match in right image (constrained by baseline)
    # # For parallel stereo, extrema should be at similar y, but different x (disparity)
    # search_x_min = half_win_width
    # search_x_max = img_right.shape[1] - half_win_width  # Right image should be to the left or same x

    # axis = []
    # for x_R in range(search_x_min, search_x_max):
    #     x1_R = x_R - half_win_width
    #     x2_R = x1_R + kernel_width
        
    #     patch_R = img_right[y1:y2, x1_R:x2_R].astype(np.float32)
        
    #     # Normalized cross-correlation (only if patches are same size and nonzero)
    #     if patch_L.shape == patch_R.shape and patch_L.std() > 0 and patch_R.std() > 0:
    #         corr = np.corrcoef(patch_L.ravel(), patch_R.ravel())[0, 1]
    #         correlation.append(corr)
    #         if corr > best_corr:
    #             best_corr = corr
    #             best_x_R = x_R
    #     axis.append(x_R)

    # Search for best match in right image (constrained by baseline, with padding)
    # For parallel stereo, extrema should be at similar y, but different x (disparity)
    search_x_min = half_win_width
    search_x_max = img_right.shape[1] + half_win_width  # Right image should be to the left or same x

    # padding
    img_right_padded = np.zeros((img_right.shape[0], img_right.shape[1] + 2*half_win_width), dtype=img_left.dtype)
    img_right_padded[:, half_win_width:half_win_width + img_right.shape[1]] = img_right
    print(img_right_padded.shape)

    axis = []
    for x_R in range(search_x_min, search_x_max):
        x_R_orig = x_R - half_win_width
        x1_R = x_R - half_win_width
        x2_R = x1_R + kernel_width
        
        patch_R = img_right_padded[y1:y2, x1_R:x2_R].astype(np.float32)
        
        # Normalized cross-correlation (only if patches are same size and nonzero)
        if patch_L.shape == patch_R.shape and patch_L.std() > 0 and patch_R.std() > 0:
            corr = np.corrcoef(patch_L.ravel(), patch_R.ravel())[0, 1]
            correlation.append(corr)
            if corr > best_corr:
                best_corr = corr
                best_x_R = x_R_orig
        axis.append(x_R_orig)
    return axis, corr, best_x_R

def correlation_2d(cam0, cam1, kernel_center, kernel_size):
    correlation = []
    kernel_height = img_left.shape[0]
    half_win_width = kernel_size[1] // 2
    half_win_height = kernel_size[0] // 2
    img_center = kernel_center

    # Extract window
    y1, y2 = 0, img_left.shape[0]
    x1 = img_center - half_win_width
    x2 = x1 + kernel_size[1]

    best_corr = -np.inf
    best_x_R = None

    # create left image patch
    patch_L = img_left[y1:y2, x1:x2].astype(np.float32)
        # --- extract 2 dimensional correlation ---
    search_x_min = half_win_width
    search_x_max = img_right.shape[1] + half_win_width  # Right image should be to the left or same x
    search_y_min = half_win_height
    search_y_max = img_right.shape[0] + half_win_height

    # padding
    img_right_padded = np.zeros((img_right.shape[0] + 2*half_win_height, img_right.shape[1] + 2*half_win_width), dtype=img_left.dtype)
    img_right_padded[half_win_height:half_win_height + img_right.shape[0], half_win_width:half_win_width + img_right.shape[1]] = img_right
    print(img_right_padded.shape)

    axis = []
    corr_map = np.zeros((img_right.shape), dtype=np.float32)

    for y_R in range(search_y_min, search_y_max):
        y_R_orig = y_R - half_win_height
        y1_R = y_R - half_win_height
        y2_R = y1_R + kernel_size[0]
        for x_R in range(search_x_min, search_x_max):
            x_R_orig = x_R - half_win_width
            x1_R = x_R - half_win_width
            x2_R = x1_R + kernel_size[1]
            
            patch_R = img_right_padded[y1_R:y2_R, x1_R:x2_R].astype(np.float32)
            
            # Normalized cross-correlation (only if patches are same size and nonzero)
            if patch_L.shape == patch_R.shape and patch_L.std() > 0 and patch_R.std() > 0:
                corr = np.corrcoef(patch_L.ravel(), patch_R.ravel())[0, 1]
                corr_map[y_R_orig,x_R_orig] = corr
                correlation.append(corr)
                if corr > best_corr:
                    best_corr = corr
                    best_x_R = x_R_orig
            axis.append(x_R_orig)
    best = np.unravel_index(np.argmax(corr_map), corr_map.shape)
    return axis, corr_map, best


# Load phase arrays
cam0 = np.load(os.path.join(input_folder,'cam0_phase.npy'))
cam1 = np.load(os.path.join(input_folder,'cam1_phase.npy'))
img_left  = to_image(cam0)
img_right = to_image(cam1)

# Tuning parameters (adjust based on your phase pattern)
size = 51                                    # Larger neighborhood = fewer extrema
threshold_mult = 1.5                         # Stricter = fewer extrema
min_distance_NMS = 80                        # Cluster suppression radius

# Find candidate extrema
left_threshold_high, left_threshold_low = compute_thresholds(img_left,threshold_mult)
local_max = (img_left == maximum_filter(img_left, size=size)) & (img_left > left_threshold_high)
local_min = (img_left == minimum_filter(img_left, size=size)) & (img_left < left_threshold_low)
extrema_candidates_L = np.argwhere(local_max | local_min)

# Suppress nearby ones
extrema_L = suppress_nearby_extrema(extrema_candidates_L, img_left, min_distance=min_distance_NMS)

# print(f"After filtering: {len(extrema_L)} extrema in the left image")

# --- Match extrema across images using cross-correlation ---
# window_size = 501                   # size of the local patch around extrema [px]
# search_range = 600                   # horizontal search range [px]
# matches = match_extrema(img_left, img_right, extrema_L, 
#                         window_size, search_range)

# print(f"Matched {len(matches)} extrema pairs")

# --- 1-dimentional correlation ---
center = (img_left.shape[1]//2)
kernel_width = img_left.shape[0]
kernel_shape = img_left.shape
#ax, corr_1d_8bit, best_x_R_8bit = correlation_1d(img_left, img_right, center, width)
# without 8-bit conversion
ax1, corr_1d, best_x_R = correlation_1d(cam0, cam1, center, kernel_width)
ax2, corr_2d, (best_y,best_x) = correlation_2d(cam0, cam1, center, kernel_shape)


#disp_8bit = center - best_x_R_8bit
disp = center - best_x_R


# --- Compute depth for each match ---
config_path = "config_stereo.json"
with open(config_path, "r") as f:
    config = json.load(f)

B = config["BOS"]["cameras_spacing"]
f_mm = config["camera"]["focal_length"]
sensor_mm = config["camera"]["sensor_size"]
W_px = config["camera"]["resolution_x"]

f_px = (f_mm / sensor_mm) * W_px
Z = (f_px * B) / disp

#print("depth with 8 bit image: ", (f_px * B) / disp_8bit)
print("depth with original image: ", (f_px * B) / disp)


# plt.figure()
# plt.plot(ax, corr_1d)
# plt.title('Image Correlation')
# plt.xlim(0, img_right.shape[1]); plt.ylim(0, 1)
# plt.show()

stats_text = (
    "At best correlation:\n"
    f"Disparity: {disp} px\n"
    f"Depth:   {Z:.3f} m\n"
)

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6))
im0 = ax0.imshow(cam0, cmap='viridis')
ax0.set_title('Left image')
ax0.set_xlim(0, img_left.shape[1]); ax0.set_ylim(img_left.shape[0], 0)
cbar0 = plt.colorbar(im0, ax=ax0)
cbar0.set_label('Intensity', rotation=270, labelpad=15)

im1 = ax1.imshow(cam1, cmap='viridis')
ax1.set_title('Right image')
ax1.set_xlim(0, img_right.shape[1]); ax1.set_ylim(img_right.shape[0], 0)
cbar0 = plt.colorbar(im1, ax=ax1)
cbar0.set_label('Intensity', rotation=270, labelpad=15)

plt.tight_layout()
plt.show(block=False)
fig.savefig(os.path.join(output_folder,'stereo.png'), dpi=fig.dpi)


plt.figure()
plt.plot(ax1, corr_1d)
plt.title('Image Correlation')
plt.xlim(0, img_right.shape[1]); 
#plt.ylim(0, 1)
plt.xlabel("px")
if config["distortions"]["turbulence_number"] == 1:
    plt.text(
        1350, 0.05, stats_text,
        fontsize=10,
        color='white',
        bbox=dict(facecolor='black', alpha=0.5, edgecolor='none')
    )
else:
    pass

plt.show(block=False)


plt.imshow(corr_2d, cmap='hot')
plt.colorbar(label='Correlation')
plt.plot(best_x,best_y, 'b+', markersize=15)
plt.show



# # Compute depth for each match
# depths = []
# for y_L, x_L, y_R, x_R, disparity, corr in matches:
#     if disparity > 0.5:  # Avoid near-zero disparities
#         Z = (f_px * B) / disparity
#         depths.append(Z)
#         print(f"Extrema at ({x_L}, {y_L}): disparity={disparity:.2f} px, depth={Z:.3f} m, corr={corr:.3f}")

# # --- Estimate phase screen position ---
# if depths:
#     median_depth = np.median(depths)
#     mean_depth = np.mean(depths)
#     std_depth = np.std(depths)
    
#     print(f"\n=== Phase Screen Localization ===")
#     print(f"Median depth: {median_depth:.3f} m")
#     print(f"Mean depth:   {mean_depth:.3f} m")
#     print(f"Std dev:      {std_depth:.3f} m")

#     # visualize matches
#     stats_text = (
#         f"Median depth: {median_depth:.3f} m\n"
#         f"Mean depth:   {mean_depth:.3f} m\n"
#         f"Std dev:      {std_depth:.3f} m"
#     )

#     fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6))
#     im0 = ax0.imshow(img_left, cmap='viridis')
#     ax0.set_title('Left image with extrema')
#     for y_L, x_L, y_R, x_R, disp, corr in matches[:10]:
#         ax0.plot(x_L, y_L, 'r+', markersize=10)
#     ax0.set_xlim(0, img_left.shape[1]); ax0.set_ylim(img_left.shape[0], 0)
#     # cbar0 = plt.colorbar(im0, ax=ax0)
#     # cbar0.set_label('Intensity', rotation=270, labelpad=15)

#     im1 = ax1.imshow(img_right, cmap='viridis')
#     ax1.set_title('Right image with matched extrema')
#     for y_L, x_L, y_R, x_R, disp, corr in matches[:10]:
#         ax1.plot(x_R, y_R, 'r+', markersize=10)
#     ax1.set_xlim(0, img_right.shape[1]); ax1.set_ylim(img_right.shape[0], 0)
#     # cbar1 = plt.colorbar(im1, ax=ax1)
#     # cbar1.set_label('Intensity', rotation=270, labelpad=15)

#     ax1.text(
#         0.02, 0.98, stats_text,
#         transform=ax1.transAxes,
#         fontsize=10,
#         color='white',
#         va='top',
#         ha='left',
#         bbox=dict(facecolor='black', alpha=0.6, edgecolor='none')
#     )

#     plt.tight_layout()
#     plt.show()
#     fig.savefig(os.path.join(output_folder,'stereo.png'), dpi=fig.dpi)
    
# else:
#     fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6))
#     ax0.imshow(img_left, cmap='viridis')
#     ax0.set_title('Left image with extrema')
#     for y_L, x_L, y_R, x_R, disp, corr in matches[:10]:
#         ax0.plot(x_L, y_L, 'r+', markersize=10)
#     ax0.set_xlim(0, img_left.shape[1]); ax0.set_ylim(img_left.shape[0], 0)

#     ax1.imshow(img_right, cmap='viridis')
#     ax1.set_title('Right image with matched extrema')
#     for y_L, x_L, y_R, x_R, disp, corr in matches[:10]:
#         ax1.plot(x_R, y_R, 'r+', markersize=10)
#     ax1.set_xlim(0, img_right.shape[1]); ax1.set_ylim(img_right.shape[0], 0)


#     plt.tight_layout()
#     plt.colorbar()
#     plt.show()
#     fig.savefig(os.path.join(output_folder,'stereo.png'), dpi=fig.dpi)
    
#     print("No valid matches found.")