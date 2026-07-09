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


def predict_eps_phase(L, d, n, n0):
    # function for generating the predicted/analytical deflection and phase
    # output in physical units

    x = np.linspace(-L/2, L/2, 2**8)
    X, Y = np.meshgrid(x, x)
    r = np.hypot(X, Y)                                      # r = sqrt(X**2 + Y**2)
    r[r == 0] = 1e-12

    theta0 = np.arctan(r / d)
    eps    = theta0 - np.arcsin((n0/n) * np.sin(theta0))   # radial deflection [rad]

    eps_x = eps * X / r
    eps_y = eps * Y / r

    a = (1/(2*d)) * (1 - n0/n)         # coefficient from the derivation
    phase = a * r**2
    phase -= phase.mean()

    return eps_x, eps_y, phase

def solidified_opd(L, N, z_A, z_D, n, n0, thickness):
    z_B = z_A + z_D
    x = np.linspace(-L/2, L/2, N)
    X, Y = np.meshgrid(x, x)
    r = np.hypot(X, Y)

    # reference path length
    theta_ref = np.arctan(r/(z_B + thickness))
    L_ref = (z_B + thickness)/np.cos(theta_ref)
    OPL_ref = n0 * L_ref

    # actual path length
    theta_0 = np.arctan(r / z_B)
    theta_1 = np.arcsin((n0/n) * np.sin(theta_0))
    L_13 = z_B/np.cos(theta_0)
    L_2 = thickness/np.cos(theta_1)
    OPL_actual = (n0 * L_13) + (n * L_2)

    OPD = OPL_actual - OPL_ref
    OPD -= OPD.mean()
    return OPD

output_folder = Path("outputs/processing_results/glass_prediction")
os.makedirs(output_folder, exist_ok=True)

# to get the expected gradient (also in pixel distance)
config_path = "config_stereo.json"
with open(config_path, "r") as f:
    config = json.load(f)
f_mm = config["camera"]["focal_length"]
sensor_mm = config["camera"]["sensor_size"]
z_A = config["distortions"]["turbulence_distance"][0]
z_B = config["BOS"]["distance_camera_screen"]
z_D = z_B - z_A

n0 = 1
n = config["refractive_index"][0]
width = ((config["camera"]["sensor_size"]/config["camera"]["focal_length"])*config["distortions"]["turbulence_distance"][0])
height = width * (config["camera"]["resolution_y"]/config["camera"]["resolution_x"]) * 0.95

L_glass    = height*2                              # physical side of the square glass [m]  <-- the only new input
print(L_glass)
eps_x, eps_y, phase_pred = predict_eps_phase(L_glass, z_D, n, n0)

# convert to pixel units
f_m  = f_mm * 1e-3                                # focal length in metres
f_px = f_mm / sensor_mm * 1290                    # focal length in pixels
S_px = f_px * z_D / (z_D + z_A - f_m)             # exact gain [px/rad]

u_pred        = S_px * eps_x
v_pred        = S_px * eps_y
phase_pred_px = S_px * phase_pred

# N = 512
# n = 1.1
# n0 = 1
# thickness = 0.002

# eps_x, eps_y, phase = eps_phase(L_glass, N, z_D, n, n0, thickness, exact=True)
# opd1 = flat_screen_opd_explicit(L_glass, 512, z_A, z_D, 1.1, 1, 0.002)
opd1 = solidified_opd(L_glass, 512, z_A, z_D, 1.1, 1, 0.002)
plt.figure()
plt.imshow(opd1)
cbar0 = plt.colorbar()
cbar0.set_label('(m)', rotation=270, labelpad=15)
# plt.show(block=False)

# fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6))
# im0 = ax0.imshow(eps_x, cmap='viridis')
# ax0.set_title('Left image')
# ax0.set_xlim(0, eps_x.shape[1]); ax0.set_ylim(eps_x.shape[0], 0)
# cbar0 = plt.colorbar(im0, ax=ax0)
# cbar0.set_label('(rad)', rotation=270, labelpad=15)

# im1 = ax1.imshow(eps_y, cmap='viridis')
# ax1.set_title('Right image')
# ax1.set_xlim(0, eps_y.shape[1]); ax1.set_ylim(eps_y.shape[0], 0)
# cbar0 = plt.colorbar(im1, ax=ax1)
# cbar0.set_label('(rad)', rotation=270, labelpad=15)

# plt.tight_layout()
plt.show()



# np.save(os.path.join(output_folder,'analytical_phase_px.npy'),phase_pred_px)
# np.save(os.path.join(output_folder,'analytical_xdisp_px.npy'),u_pred)
# np.save(os.path.join(output_folder,'analytical_ydisp_px.npy'),v_pred)
np.save(os.path.join(output_folder,'analytical_phase.npy'),phase_pred)
np.save(os.path.join(output_folder,'analytical_xdisp.npy'),eps_x)
np.save(os.path.join(output_folder,'analytical_ydisp.npy'),eps_y)