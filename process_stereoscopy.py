import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter, minimum_filter
from scipy.ndimage import label
from scipy.optimize import linear_sum_assignment
from pathlib import Path
import glob, os
from scipy import signal, datasets, ndimage

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
    return axis, correlation, best_x_R

def correlation_2d(cam0, cam1, kernel_center, kernel_size):

    cam0_flipped = cam0[::-1,::-1]
    corr_scipy = signal.fftconvolve(cam1, cam0_flipped,mode='same')
    best = np.unravel_index(np.argmax(corr_scipy), corr_scipy.shape)

    return corr_scipy, best

def masking(img, threshold_frac=0.1, min_region_size=500):
    # """
    # Create a binary mask marking pixels with significant phase content.
    
    # Parameters
    # ----------
    # phase : 2D array
    #     Reconstructed phase map.
    # threshold_frac : float
    #     Threshold as a fraction of max |phase|.
    # min_region_size : int
    #     Minimum connected region size (pixels) to keep.
    
    # Returns
    # -------
    # mask : 2D boolean array
    # """
    # # Absolute threshold on phase magnitude
    # threshold = threshold_frac * np.abs(img).max()
    # # threshold = 1e-4
    # mask = np.abs(img) > threshold
    
    # # # Optional: remove small isolated pixels (noise)
    # # from scipy.ndimage import binary_opening, label
    # # mask = binary_opening(mask, iterations=2)
    
    # # Optional: keep only large connected regions
    # labeled, num = label(mask)
    # for i in range(1, num + 1):
    #     if np.sum(labeled == i) < min_region_size:
    #         mask[labeled == i] = False

    return mask

# def magnitude_optical_flow(u_L, v_L, u_R, v_R):
#     # --- this function takes the optical flow of left and right images as input and uses optical flow to see the correlation ---

#     return magnitude


# Load phase arrays
cam0 = np.load(os.path.join(input_folder,'cam0_phase.npy'))
cam1 = np.load(os.path.join(input_folder,'cam1_phase.npy'))
img_left  = to_image(cam0)
img_right = to_image(cam1)

# # Very rough: crop 100 pixels from each side (whatever your empty region width is)
# crop = 500
# pL_cropped = cam0[:, crop:]
# pR_cropped = cam1[:, :-crop]
# print(pL_cropped.shape)
# print(pR_cropped.shape)

# # Now correlate the cropped maps
# H, W_new = pL_cropped.shape
# C = np.zeros(2*W_new - 1)

# fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6))
# im0 = ax0.imshow(pL_cropped, cmap='viridis')
# ax0.set_title('Left image')
# ax0.set_xlim(0, pL_cropped.shape[1]); ax0.set_ylim(pL_cropped.shape[0], 0)
# cbar0 = plt.colorbar(im0, ax=ax0)
# cbar0.set_label('Intensity (m)', rotation=270, labelpad=15)

# im1 = ax1.imshow(pR_cropped, cmap='viridis')
# ax1.set_title('Right image')
# ax1.set_xlim(0, pR_cropped.shape[1]); ax1.set_ylim(pR_cropped.shape[0], 0)
# cbar0 = plt.colorbar(im1, ax=ax1)
# cbar0.set_label('Intensity (m)', rotation=270, labelpad=15)

# plt.tight_layout()
# plt.show(block=False)

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

# # mask the phase images before correlation
# mask_L = masking(cam0)
# mask_R = masking(cam1)
# fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6))
# im0 = ax0.imshow(mask_L)
# ax0.set_title('Left image')
# ax0.set_xlim(0, img_left.shape[1]); ax0.set_ylim(img_left.shape[0], 0)
# cbar0 = plt.colorbar(im0, ax=ax0)
# cbar0.set_label('Intensity (m)', rotation=270, labelpad=15)

# im1 = ax1.imshow(mask_R)
# ax1.set_title('Right image')
# ax1.set_xlim(0, img_right.shape[1]); ax1.set_ylim(img_right.shape[0], 0)
# cbar0 = plt.colorbar(im1, ax=ax1)
# cbar0.set_label('Intensity (m)', rotation=270, labelpad=15)

# plt.tight_layout()
# plt.show(block=False)

# --- 1-dimentional correlation ---
center = (img_left.shape[1]//2)
kernel_width = img_left.shape[1]
kernel_shape = img_left.shape
#ax, corr_1d_8bit, best_x_R_8bit = correlation_1d(img_left, img_right, center, width)
# without 8-bit conversion
ax, corr_1d, best_x_R = correlation_1d(cam0, cam1, center, kernel_width)
# ax2, corr_2d, (best_y,best_x) = correlation_2d(cam0, cam1, center, kernel_shape)
corr_2d, (best_y,best_x) = correlation_2d(cam0, cam1, center, kernel_shape)

# center1 = pL_cropped.shape[1]//2
# kernel_width1 = pL_cropped.shape[1]
# ax_1, corr_1d1, best = correlation_1d(pL_cropped, pR_cropped, center1, kernel_width1)



#disp_8bit = center - best_x_R_8bit
disp = center - best_x_R
disp_fft = center - best_x
print(f"disparity from fftconvolve: {disp_fft}px\n")


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

# Configuration of the experimental setup
B_exp = 0.06        # 6 cm
f_mm_exp = 2.8      # computed from HFOV and sensor dimensions
sensor_mm_exp = 3.4 # camera specification
W_px_exp = 1280     # width resolution

f_px_exp = (f_mm_exp / sensor_mm_exp) * W_px_exp
Z_exp = (f_px_exp * B_exp) / disp


#print("depth with 8 bit image: ", (f_px * B) / disp_8bit)
print("depth with original image: ", (f_px * B) / disp)
print("depth with original image (from fftconvolve): ", (f_px * B) / disp_fft)
# print("depth with original image: ", (f_px_exp * B_exp) / disp)
# print("depth with original image (from fftconvolve): ", (f_px_exp * B_exp) / disp_fft)

# plt.figure()
# plt.plot(ax, corr_1d)
# plt.title('Image Correlation')
# plt.xlim(0, img_right.shape[1]); plt.ylim(0, 1)
# plt.show()

# Using optical flow data


# --- visualize ---

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6))
im0 = ax0.imshow(cam0, cmap='viridis')
ax0.set_title('Left image')
ax0.set_xlim(0, img_left.shape[1]); ax0.set_ylim(img_left.shape[0], 0)
cbar0 = plt.colorbar(im0, ax=ax0)
cbar0.set_label('Intensity (m)', rotation=270, labelpad=15)

im1 = ax1.imshow(cam1, cmap='viridis')
ax1.set_title('Right image')
ax1.set_xlim(0, img_right.shape[1]); ax1.set_ylim(img_right.shape[0], 0)
cbar0 = plt.colorbar(im1, ax=ax1)
cbar0.set_label('Intensity (m)', rotation=270, labelpad=15)

plt.tight_layout()
plt.show(block=False)
# fig.savefig(os.path.join(output_folder,'stereo.png'), dpi=fig.dpi)
fname = "7_7.png"
# fig.savefig(f"../Stereo/29062026/correlation_maps/turbulent/double_screen/images/{fname}", dpi=fig.dpi)

stats_text = (
    "At best correlation:\n"
    f"Disparity: {disp} px\n"
    f"Depth:   {Z:.3f} m\n"
)

plt.figure()
plt.plot(ax, corr_1d)
plt.title('Image Correlation')
plt.xlim(0, img_right.shape[1])
#plt.ylim(0, 1)
plt.xlabel("px")
if config["distortions"]["turbulence_number"] == 1:
    plt.text(
        150, 0.05, stats_text,
        fontsize=10,
        color='white',
        bbox=dict(facecolor='black', alpha=0.5, edgecolor='none')
    )
else:
    pass
plt.show(block=False)

plt.figure()
plt.imshow(corr_2d, cmap='hot')
plt.colorbar(label='Correlation')
plt.plot(best_x,best_y, 'b+', markersize=15)
plt.text(
    best_x + 2, best_y + 2, 
    f'({best_x}, {best_y})', 
    color='white', 
    weight='bold', 
    fontsize=10,
    bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.3')
)
# plt.savefig(f"../Stereo/29062026/correlation_maps/turbulent/double_screen/different/{fname}", dpi=plt.gcf().dpi)
# plt.savefig("../Stereo/29062026/correlation_maps/turbulent/double_screen/same/1_1.png", dpi=plt.gcf().dpi)
plt.show(block=False)


plt.figure()
corr1d_from_2d = corr_2d[540,:]
plt.plot(corr1d_from_2d)
plt.title('Correlation at the middle')
plt.xlim(0, img_right.shape[1])
# plt.show(block=False)
plt.show()

# print(len(corr1d_from_2d))
# Next thing is to get the two peaks and get their disparities respectively
# peaks, props = signal.find_peaks(corr1d_from_2d, prominence=0.05*np.max(np.abs(corr1d_from_2d)), distance=20)

peaks, _ = signal.find_peaks(corr1d_from_2d)

# # Take the two peaks with the highest correlation values
# top_two = peaks[np.argsort(corr1d_from_2d[peaks])[-2:]]
# # Sort by prominence, keep top 2
# # order = np.argsort(props['prominences'])[::-1][:2]
# # top_two_peaks = peaks[order]

# # Get disparities
# center = len(corr1d_from_2d) // 2
# disparities = np.abs(top_two - center)

# # Convert to depths
# depths = f_px * B / disparities

# print(f"Top two peaks: {top_two[0]}, {top_two[1]}\n")
# print(f"Disparities: {disparities[0]}, {disparities[1]}\n")
# print(f"Predicted depths: {depths[0]}, {depths[1]}\n")

# plt.figure()
# plt.plot(ax_1, corr_1d1)
# plt.title('Cropped Image Correlation')
# plt.xlim(0, pL_cropped.shape[1])
# #plt.ylim(0, 1)
# plt.show()