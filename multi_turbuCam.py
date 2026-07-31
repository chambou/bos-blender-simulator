# Script to process multiple turbulent images with one reference

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
from scipy.fft import dctn, idctn
from process_turbuCam import reconstruct_from_gradient

def load_image(path):
    data = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    print(data.dtype, data.min(), data.max(), data.shape)
    data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    print(data.dtype, data.min(), data.max(), data.shape)
    return data

# input_folder = Path("../experimental/28072026/variable_distance/images_1")
input_folder = Path('../experimental/31072026/multiple_frames/captured_frames_tif_3/left_images') 

img = "left"

# Camera properties
B = 0.06        # 6 cm
f_mm = 40             # camera specification
sensor_mm = 3.84        # camera specification
W_px = 1280     # width resolution

if img == "left":
    img_code = 0
    print("Processing left camera images...")
else:
    img_code = 1
    print("Processing right camera images...")

imgs = glob.glob(os.path.join(input_folder, f"*_{img_code}_*.tif")) 
print(imgs)
N_imgs = len(imgs)  # how many images
print(f"Found {N_imgs} turbulent image/s...")

ref_path = glob.glob(os.path.join(input_folder, f"*_ref_*_{img_code}*.tif"))[0]

blender_used = False

max_deflections = []
max_displacements = []
sensitivities = []
orig_frames = []    # video mode
frames = []         # video mode

# z_D = [0.2, 0.4, 0.6, 0.8, 1, 1.2]
z_D = 0.2
z_B = 1.50       # 0.6 usually
for k in range(0,N_imgs):

    # Configuration of the experimental setup
    # because distances change on every trial
    z_A = z_B - z_D
    # z_D = z_B - z_A
    f_m = f_mm * 1e-3                                # focal length in metres
    f_px = (f_mm / sensor_mm) * W_px                # focal length in pixels
    pitch = (sensor_mm / W_px) * 1e-3                   # [m/px]
    S_m = f_m * z_D / (z_B - f_m)                 # Sensitivity

    sensitivities.append(S_m)

    ext = f"*_{img_code}_"+str(k)+".tif"
    img_now = glob.glob(os.path.join(input_folder, ext))[0]
    print("image now: ", img_now)

    ref_frame = load_image(ref_path)[...,0]
    print('ref frame:', ref_frame.shape)
    turbu_frame = load_image(img_now)
    orig_frames.append(turbu_frame)

    flow = cv2.calcOpticalFlowFarneback(
        ref_frame,turbu_frame , None,
        pyr_scale=0.5, levels=10, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )

    u, v = flow[..., 0], flow[..., 1] # Displacement in X and Y direction in the sensor [px]

    phase_px = reconstruct_from_gradient(u, v)

    delta_x = u * pitch
    delta_y = v * pitch
    delta_mag = np.sqrt(u**2 + v**2) * pitch        # displacement in the sensor
    max_delta = delta_mag.max()
    max_displacements.append(max_delta)

    eps_x = delta_x / S_m
    eps_y = delta_y / S_m
    deflection_mag = np.sqrt(delta_x**2 + delta_y**2) / S_m
    max_deflection = deflection_mag.max()
    max_deflections.append(max_deflection * 1e3)        # mrad

    psi_screen = z_A / f_px
    opd = reconstruct_from_gradient(eps_x, eps_y) * psi_screen
    frames.append(opd)

    # magnitude = np.sqrt(u**2 + v**2)
    # magnitude = u**2 + v**2

    # plt.figure(figsize=(8, 6))
    # plt.imshow(deflection_mag, cmap='hot')
    # plt.colorbar(label='Deflection magnitude [rad]')
    # plt.title('Optical Flow Magnitude')
    # # Is this the last iteration?
    # is_last = (k == N_imgs - 1)

    # if is_last:
    #     plt.show()  # Blocks and keeps the window open at the end
    # else:
    #     plt.show(block=False)
    #     plt.pause(0.5)  # Pause briefly so the GUI window updates
    # plt.show(block=False)


    print(k, " is finished")
    # z_D += 0.2

mean_deflection = np.mean(max_deflections)
std_deflection = np.std(max_deflections)
coeff_var = std_deflection / mean_deflection * 100

print("Mean eps:", mean_deflection)
print("Std eps:", std_deflection)
print("Variation:", coeff_var)
# print(frames.shape)

# stats_text = (
#     f"Mean: {mean_deflection:.2f}\n" f"Std Dev: {std_deflection:.2f}\n"
# )

# x = [20, 40, 60, 80, 100, 120]

# plt.figure()
# plt.plot(x, max_displacements, marker="o", color="b", linestyle="-")
# plt.title("Maximum Displacements in the sensor")
# plt.xlabel("Distance [cm]")
# plt.ylabel("Magnitude [m]")
# plt.grid(True)
# plt.show(block=False)

# plt.figure()
# plt.plot(x, max_deflections, marker="o", color="b", linestyle="-")
# plt.axhline(
#     mean_deflection,
#     color="r",
#     linestyle="--",
#     linewidth=1.5,
#     label=f"Mean ({mean_deflection:.1f})",
# )
# # transform=plt.gca().transAxes places text using relative coordinates:
# # (x=0.05 is 5% from left, y=0.95 is 95% from bottom)
# plt.gca().text(
#     0.75,
#     0.95,
#     stats_text,
#     transform=plt.gca().transAxes,
#     fontsize=10,
#     verticalalignment="top",
#     bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8, edgecolor="gray")
# )
# plt.title("Maximum Deflections")
# plt.xlabel("Distance [cm]")
# plt.ylabel("Magnitude [mrad]")
# plt.grid(True)
# plt.show()

# --- Video creation (side by side) ---

# cmap = plt.get_cmap('viridis')
# output_video = "output_movie4.mp4"
# fps = 30

# height, width = frames[0].shape[:2]
# width = width * 2
# fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# # Set isColor=False for single-channel 2D images
# out = cv2.VideoWriter(
#     output_video, fourcc, fps, (width, height), isColor=True
# )

# for frame1, frame2 in zip(orig_frames, frames):
#     orig = cv2.normalize(frame1, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U) if frame1.dtype != "uint8" else frame1
#     frame_8u = cv2.normalize(frame2, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U) if frame2.dtype != "uint8" else frame2
#     # out.write(frame_8u)

#     # 3 channel convertion
#     f1_rgb = cv2.cvtColor(orig, cv2.COLOR_GRAY2RGB)
#     frame_8u = 255 - frame_8u
#     colored_frame = cv2.applyColorMap(frame_8u, cv2.COLORMAP_VIRIDIS)

#     combined = np.hstack((f1_rgb, colored_frame))

#     out.write(combined)

# out.release()
# print(f"Saved movie to {output_video}")