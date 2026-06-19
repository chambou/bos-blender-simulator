import cv2
import numpy as np
from astropy.io import fits
import glob, os
from tqdm import trange
import matplotlib.pyplot as plt
import tifffile as tiff
from scipy.fft import fft2, ifft2
from pathlib import Path
import json
from PIL import Image

# --- This code is to process stereo images of phase screens and localize them ---

def to_image(array):
    # Find min and max ignoring NaNs
    amin, amax = np.nanmin(array), np.nanmax(array)
    
    # Check if the array is completely uniform to prevent division by zero
    if amax == amin:
        return np.zeros(array.shape, dtype=np.uint8)
        
    # Safely normalize
    norm = (array - amin) / (amax - amin)
    return (255 * norm).astype(np.uint8)

# load files
cam0 = np.load('outputs/processing_results/cam0_phase.npy')
cam1 = np.load('outputs/processing_results/cam1_phase.npy')

# convert numpy array to images
img_left    = to_image(cam0)
img_right   = to_image(cam1)

plt.figure()
plt.imshow(img_left)
plt.colorbar()
plt.show(block=False)

plt.figure()
plt.imshow(img_right)
plt.colorbar()
plt.show()

# # creates StereoBm object 
# stereo = cv2.StereoBM_create(numDisparities = 16,
#                             blockSize = 15)

# # computes disparity
# disparity = stereo.compute(img_right, img_left)

# # displays image as grayscale and plotted
# plt.imshow(disparity, 'gray')
# plt.show()

# Create the Stereo Matcher object
# You may need to tune these parameters depending on your camera resolution
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=64,  # Must be divisible by 16
    blockSize=11,
    P1=8 * 3 * 11**2,
    P2=32 * 3 * 11**2,
    disp12MaxDiff=1,
    uniquenessRatio=15,
    speckleWindowSize=100,
    speckleRange=2
)

# 3. Compute the disparity
# OpenCV outputs a 16-bit signed integer array, so we divide by 16 to get actual pixel shifts
disparity = stereo.compute(img_left, img_right).astype(np.float32) / 16.0

# displays image as grayscale and plotted
plt.imshow(disparity, 'gray')
plt.show()

# Camera Parameters (You must know these from your hardware setup)
# conversion to physical units
config_path = "config_stereo.json"
with open(config_path, "r") as f:
    config = json.load(f)
B = config["BOS"]["cameras_spacing"]
f_mm = config["camera"]["focal_length"]
sensor_mm = config["camera"]["sensor_size"]
z_A = config["distortions"]["turbulence_distance"][0]
z_B = config["BOS"]["distance_camera_screen"]
z_D = z_B - z_A

f_m  = f_mm * 1e-3                                # focal length in metres
f_px = f_mm / sensor_mm * 1290                    # focal length in pixels
S_px = f_px * z_D / (z_D + z_A - f_m)             # exact gain [px/rad]
psi_screen = z_A / f_px                           # screen sampling [m/px]

focal_length_pixels = f_px  # Example focal length in pixels
baseline_meters = B        # Example distance between cameras (12 cm)

# Target pixel coordinates of the object (e.g., center of a bounding box)
x, y = 200, 150 

# Get the disparity value at that specific point
disp_value = disparity[y, x]

# Prevent division by zero for objects that are infinitely far away
if disp_value > 0:
    # Apply the formula: Depth = (f * B) / d
    distance = (focal_length_pixels * baseline_meters) / disp_value
    print(f"Distance to object: {distance:.2f} meters")
else:
    print("Object is too far away or depth cannot be computed.")