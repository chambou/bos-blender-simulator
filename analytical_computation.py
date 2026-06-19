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

output_folder = Path("outputs/processing_results")
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
n = config["refractive_index"]
L_glass    = 0.2                              # physical side of the square glass [m]  <-- the only new input

eps_x, eps_y, phase_pred = predict_eps_phase(L_glass, z_D, n, n0)

# convert to pixel units
f_m  = f_mm * 1e-3                                # focal length in metres
f_px = f_mm / sensor_mm * 1290                    # focal length in pixels
S_px = f_px * z_D / (z_D + z_A - f_m)             # exact gain [px/rad]

u_pred        = S_px * eps_x
v_pred        = S_px * eps_y
phase_pred_px = S_px * phase_pred

np.save(os.path.join(output_folder,'analytical_phase_px.npy'),phase_pred_px)
np.save(os.path.join(output_folder,'analytical_xdisp_px.npy'),u_pred)
np.save(os.path.join(output_folder,'analytical_ydisp_px.npy'),v_pred)
np.save(os.path.join(output_folder,'analytical_phase.npy'),phase_pred)
np.save(os.path.join(output_folder,'analytical_xdisp.npy'),eps_x)
np.save(os.path.join(output_folder,'analytical_ydisp.npy'),eps_y)