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


def predict_eps_opd(L, d, n, n0):
    # function for generating the predicted/analytical deflection and phase
    # output in physical units

    x = np.linspace(-L/2, L/2, 1080)
    X, Y = np.meshgrid(x, x)
    r = np.hypot(X, Y)                                      # r = sqrt(X**2 + Y**2)
    r[r == 0] = 1e-12

    theta0 = np.arctan(r / d)
    theta1 = np.arcsin((n0/n) * np.sin(theta0))
    eps    = theta0 - theta1   # radial deflection [rad]

    eps_x = eps * X / r
    eps_y = eps * Y / r

    a = (n0/(2*d)) * (1 - n0/n)         # coefficient from the derivation
    opd = a * r**2
    opd -= opd.mean()

    # analytic_normalized = opd / np.abs(opd).max()
    # plt.figure(figsize=(8, 6))
    # plt.plot(r, analytic_normalized)
    # plt.title("Analytic Normalized")
    # print("r max: ", r.max())
    # np.save('outputs/analytical_single_int_opd_normalized.npy',analytic_normalized)

    return eps_x, eps_y, opd

def solidified_opd(L, N, z_A, z_D, n, n0, thickness):
    # half_thick = thickness / 2
    # z_A = z_A - half_thick
    # z_D = z_D - half_thick
    # z_B = z_A + z_D
    # # print(z_B)
    # x = np.linspace(-L/2, L/2, N)
    # X, Y = np.meshgrid(x, x)
    # r = np.hypot(X, Y)

    # # reference path length
    # theta_ref = np.arctan(r/(z_B + thickness))
    # L_ref = (z_B + thickness)/np.cos(theta_ref)
    # OPL_ref = n0 * L_ref

    # # actual path length
    # theta_0 = np.arctan(r / (z_B))
    # theta_1 = np.arcsin((n0/n) * np.sin(theta_0))
    # L_13 = z_B/np.cos(theta_0)
    # L_2 = thickness/np.cos(theta_1)
    # OPL_actual = (n0 * L_13) + (n * L_2)

    # OPD = OPL_actual - OPL_ref
    # OPD -= OPD.mean()

    z_B = z_A + z_D

    # x = np.linspace(-L/2, L/2, N)
    # X, Y = np.meshgrid(x, x)
    # r = np.hypot(X, Y)
    x = np.linspace(-L/2, L/2, 1080)
    X, Y = np.meshgrid(x, x)
    r = np.hypot(X, Y)                                      # r = sqrt(X**2 + Y**2)


    theta_0 = np.arctan(r / z_B)
    theta_1 = np.arcsin((n0/n) * np.sin(theta_0))

    theta_ref = theta_0

    L_13 = (z_B - thickness) / np.cos(theta_0)
    L_2 = thickness / np.cos(theta_1)
    OPL_actual = n0 * L_13 + n * L_2

    L_ref = z_B / np.cos(theta_ref)
    OPL_ref = n0 * L_ref

    opd = OPL_actual - OPL_ref
    opd -= opd.mean()

    analytic_normalized = opd / np.max(opd)
    print(analytic_normalized.max())
    plt.figure(figsize=(8, 6))
    plt.plot(r, analytic_normalized)
    plt.title("Analytic Normalized")
    print("r max: ", r.max())
    np.save('outputs/analytical_double_int_opd_normalized.npy',analytic_normalized)
    return opd

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
# n = config["refractive_index"][0]
n = 1.1
width = ((config["camera"]["sensor_size"]/config["camera"]["focal_length"])*z_A)
height = width * (config["camera"]["resolution_y"]/config["camera"]["resolution_x"])
print("width: ", width)
print("height: ", height)

L_glass    = height                             # physical side of the square glass [m]
print(L_glass)
eps_x, eps_y, opd_pred = predict_eps_opd(L_glass, z_D, n, n0)

# convert to pixel units
f_m  = f_mm * 1e-3                                # focal length in metres
f_px = f_mm / sensor_mm * 1290                    # focal length in pixels
S_px = f_px * z_D / (z_D + z_A - f_m)             # exact gain [px/rad]

# u_pred        = S_px * eps_x
# v_pred        = S_px * eps_y
# opd_pred_px = S_px * opd_pred

N = 1080
# n = 1.1
# n0 = 1
thickness = config["distortions"]["thickness"]
print(thickness)

opd = solidified_opd(L_glass, N, z_A, z_D, n, n0, thickness)


# eps_x, eps_y, phase = eps_phase(L_glass, N, z_D, n, n0, thickness, exact=True)
# opd1 = flat_screen_opd_explicit(L_glass, 512, z_A, z_D, 1.1, 1, 0.002)
# opd1 = solidified_opd(L_glass, 512, z_A, z_D, 1.5, 1, 0.018)
# plt.figure()
# plt.imshow(opd1)
# cbar0 = plt.colorbar()
# cbar0.set_label('(m)', rotation=270, labelpad=15)
# plt.show(block=False)

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6))
im0 = ax0.imshow(eps_x, cmap='viridis')
ax0.set_title('x-axis deflection')
ax0.set_xlim(0, eps_x.shape[1]); ax0.set_ylim(eps_x.shape[0], 0)
cbar0 = plt.colorbar(im0, ax=ax0)
cbar0.set_label('(rad)', rotation=270, labelpad=15)

im1 = ax1.imshow(eps_y, cmap='viridis')
ax1.set_title('y-axis deflection')
ax1.set_xlim(0, eps_y.shape[1]); ax1.set_ylim(eps_y.shape[0], 0)
cbar0 = plt.colorbar(im1, ax=ax1)
cbar0.set_label('(rad)', rotation=270, labelpad=15)

plt.tight_layout()
plt.show(block=False)

# plot magnitude
magnitude_eps = np.sqrt(eps_x**2 + eps_y**2)
plt.figure(figsize=(8, 6))
plt.imshow(magnitude_eps, cmap='viridis')
cbar = plt.colorbar()
tick_locs = np.linspace(magnitude_eps.min(), magnitude_eps.max(), 6)
cbar.set_label('Deflection Magnitude [rad]')
cbar.set_ticks(tick_locs)
# plt.title('Optical Flow Magnit?')
plt.show(block=False)

plt.figure(figsize=(8, 6))
plt.imshow(opd_pred, cmap='viridis')
cbar = plt.colorbar()
tick_locs = np.linspace(opd_pred.min(), opd_pred.max(), 6)
cbar.set_label('Optical Path Difference (OPD) [m]')
cbar.set_ticks(tick_locs)
# plt.title('Optical Flow Magnit?')
plt.show(block=False)

plt.figure(figsize=(8, 6))
plt.imshow(opd, cmap='viridis')
cbar = plt.colorbar()
tick_locs = np.linspace(opd.min(), opd.max(), 6)
cbar.set_label('Optical Path Difference (OPD) [m]')
cbar.set_ticks(tick_locs)
# plt.title('Optical Flow Magnit?')
plt.show()



# np.save(os.path.join(output_folder,'analytical_phase_px.npy'),phase_pred_px)
# np.save(os.path.join(output_folder,'analytical_xdisp_px.npy'),u_pred)
# np.save(os.path.join(output_folder,'analytical_ydisp_px.npy'),v_pred)
np.save(os.path.join(output_folder,'analytical_phase.npy'),opd_pred)
np.save(os.path.join(output_folder,'analytical_xdisp.npy'),eps_x)
np.save(os.path.join(output_folder,'analytical_ydisp.npy'),eps_y)