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

def reconstruct_from_gradient(gx, gy):
    """
    Solve Poisson equation
    """
    H, W = gx.shape

    # divergence
    div = np.zeros_like(gx)
    div[:, :-1] += gx[:, :-1]
    div[:, 1:]  -= gx[:, :-1]
    div[:-1, :] += gy[:-1, :]
    div[1:, :]  -= gy[:-1, :]

    # FFT frequencies
    yy, xx = np.meshgrid(np.fft.fftfreq(H), np.fft.fftfreq(W), indexing='ij')
    denom = (2*np.cos(2*np.pi*xx) - 2) + (2*np.cos(2*np.pi*yy) - 2)
    denom[0,0] = 1  # avoid division by zero

    f = np.real(ifft2(fft2(div) / denom))
    f -= f.mean()  # remove arbitrary constant

    return f


def predict_eps_phase(L, d, n, n0):

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

# ------------------------------
# CONFIGURATION
# ------------------------------

input_folder = Path("outputs/render_results")
output_folder = Path("outputs/processing_results")
reference_mode = "first"   # options: "median", "first", "previous"
save_fits_cube = False       # save full cube as FITS file - takes time!
show_animation = False       # show results as an animation
refresh_delay = 0.05         # seconds between frames (e.g. 0.2 = 5 FPS)
os.makedirs(output_folder, exist_ok=True)
Ncams = len(glob.glob(os.path.join(input_folder, "render_ref_*.tif")))  # how many cameras used
print(f"Found {Ncams} camera/s...")

for k in range(0,Ncams):

    ext = "_"+str(k)+".tif"
    # ------------------------------
    # LOAD IMAGE LIST
    # ------------------------------
    files = sorted(glob.glob(os.path.join(input_folder, '*'+ext)))
    if len(files) < 2:
        raise ValueError("Need at least 2 images for this camera in the folder!")

    print(f"Found {len(files)} files.")

    # ------------------------------
    # LOAD FITS IMAGE
    # ------------------------------
    def load_image(path):
        data = cv2.imread(path)
        # Normalize to 0–255 and convert to uint8 for optical flow
        norm = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))
        return (255 * norm).astype(np.uint8)

    # ------------------------------
    # LOAD ALL FRAMES
    # ------------------------------
    frames = [load_image(f) for f in files]
    frames = np.array(frames)[...,0]
    #frames[frames > 10] = 255
    #frames[frames != 255] = 0
    mask = np.ones_like(frames[0])  # mask for dome

    # ------------------------------
    # SELECT REFERENCE STRATEGY
    # ------------------------------
    if reference_mode == "median":
        reference_frame = np.median(frames, axis=0).astype(np.uint8)
    elif reference_mode == "first":
        reference_frame = frames[-1,...]
    elif reference_mode == "previous":
        reference_frame = None  # handled dynamically in loop
    else:
        raise ValueError("Invalid reference_mode. Use 'median', 'first', or 'previous'.")

    # Compute optical flow
    flow = cv2.calcOpticalFlowFarneback(
        frames[0,...],frames[1,...], None,
        pyr_scale=0.5, levels=10, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    
    u, v = flow[..., 0], flow[..., 1] # Displacement in X and Y direction
    u = -np.fliplr(u)     # mirror x AND negate (flip reverses the x-derivative)
    v =  np.fliplr(v)     # mirror x; y-derivative unchanged

    phase = reconstruct_from_gradient(u, v)


    # conversion to physical units
    config_path = "config_stereo.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    f_mm = config["camera"]["focal_length"]
    sensor_mm = config["camera"]["sensor_size"]
    z_A = config["distortions"]["turbulence_distance"][0]
    z_B = config["BOS"]["distance_camera_screen"]
    z_D = z_B - z_A

    f_m  = f_mm * 1e-3                                # focal length in metres
    f_px = f_mm / sensor_mm * 1290                    # focal length in pixels
    S_px = f_px * z_D / (z_D + z_A - f_m)             # exact gain [px/rad]

    # deflection: pixels -> radians
    eps_x = u / S_px
    eps_y = v / S_px

    # phase: pixel-integrated -> radians
    phase_rad = phase / S_px
    

    np.save(os.path.join(output_folder,'cam'+str(k)+'_phase.npy'),phase_rad)
    np.save(os.path.join(output_folder,'cam'+str(k)+'_xdisp.npy'),eps_x)
    np.save(os.path.join(output_folder,'cam'+str(k)+'_ydisp.npy'),eps_y)