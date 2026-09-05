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

def load_image(path):
    data = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    print(data.dtype, data.min(), data.max(), data.shape)
    data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    print(data.dtype, data.min(), data.max(), data.shape)
    return data

def propagate_uncertainty(z_A, z_D, f, sigma_pos, sigma_delta_px, delta_px):
    # uncertainty propagation from u,v to OPD
    # for scaled model setup

    z_B = z_A + z_D
    sigma_z_B = np.sqrt(2) * sigma_pos

    S_m = (f * z_D) / (z_B - f)
    dSm_dzA = (-f * z_D) / (z_B - f)**2
    dSm_dzD = (f * (z_A - f)) / (z_B - f)**2
    sigma_S_m = np.sqrt((dSm_dzA * sigma_pos)**2 + (dSm_dzD * sigma_pos)**2)
    rel_S_m = sigma_S_m / S_m

    rel_sigma_delta_px = sigma_delta_px / delta_px
    rel_eps = np.sqrt(rel_sigma_delta_px**2 + rel_S_m**2)

    rel_psi = sigma_pos / z_A
    rel_OPD = np.sqrt(rel_eps**2 + rel_psi**2)

 
    return rel_S_m, rel_eps, rel_OPD

def opd_to_temperature(opd):
    # function to predict temperature from opd data
    # constants
    t = 0.03       #physical path length, approximation [m]
    n_0 = 1.000293  #reference refractive index, for calibration
    p = 101325      #ambient pressure, approximation
    T_0 = 273.15    #temperature at n_0, for calibration, [K]
    R = 8.3145      #gas constant

    #molar refractivity
    A = R * (n_0**2 - 1) * T_0 / (3 * p)

    ## if reducing offset 
    # T_room_K = 24 + T_0
    # n_expected_ambient = np.sqrt((3*A*p) / (R*T_room_K) + 1)
    # OPD_expected_ambient = (n_expected_ambient - n_0) * t

    # OPD_corrected = opd - (opd.mean() - OPD_expected_ambient)

    # n_field = n_0 + OPD_corrected / t
    # T = (3*A*p) / (R* (n_field**2 -1))

    #OPD to n
    n = n_0 + (opd / t)

    # n to T
    T = (3 * A * p) / (R * (n**2 - 1))

    return T

def plot_flow_vectors(u, v, step, scale=None, background=None):
    H, W = u.shape
    y, x = np.mgrid[0:H:step, 0:W:step]
    u_sub = u[::step, ::step]
    v_sub = v[::step, ::step]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    if background is not None:
        im = ax.imshow(background, cmap='gray', alpha=0.5)
        # plt.colorbar(im, ax=ax, label='Displacement magnitude [px]')
    
    ax.quiver(x, y, u_sub, v_sub, color='red', scale=scale, 
              width=0.004, headwidth=4)
    
    ax.set_xlabel('x [px]')
    ax.set_ylabel('y [px]')
    ax.invert_yaxis()   # image-style: y increases downward
    ax.set_aspect('equal')
    plt.tight_layout()
    return fig

    
# ------------------------------
# CONFIGURATION
# ------------------------------

# specify if blender is used
blender_used = True

if blender_used:
    print("Using Blender setup")
    input_folder = Path("outputs/render_results")
    # input_folder = Path("../experimental/23072026/hamamatsu/turb1/img")
else:
    print("Using experimental setup")
    # input_folder = Path("outputs/experiments")
    # input_folder = Path("outputs/render_results")
    # input_folder = Path("../experimental/23072026/hamamatsu/turb1/img")
    input_folder = Path("../experimental/27072026/stereo_40mm/sample_2/img")
output_folder = Path("outputs/processing_results")
reference_mode = "first"   # options: "median", "first", "previous"
save_fits_cube = False       # save full cube as FITS file - takes time!
show_animation = False       # show results as an animation
refresh_delay = 0.05         # seconds between frames (e.g. 0.2 = 5 FPS)
os.makedirs(output_folder, exist_ok=True)
Ncams = len(glob.glob(os.path.join(input_folder, "*_ref_*.tif")))  # how many cameras used
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
    print(f"Now processing {files}")

    # ------------------------------
    # LOAD ALL FRAMES
    # ------------------------------
    frames = [load_image(f) for f in files]
    if blender_used == True:
        frames = np.array(frames)[...,0]        # because blender render is not single channel
    else:
        frames = np.array(frames)[...]          # if already one channel
    print(frames.shape)
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

    # --- new_flow
    # flow = cv2.calcOpticalFlowFarneback(
    #     frames[0,...], frames[1,...], None,
    #     pyr_scale=0.5, 
    #     levels=3,                            # Lower: to preserve small details
    #     winsize=5,                           # Lower: to catch smaller shifts
    #     iterations=7,                        # Increased: help resolve fast jumps
    #     poly_n=5,
    #     poly_sigma=1.2,
    #     flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN # Upgraded: Better mathematical precision for edges
    # )

    # --- new_flow1
    # flow = cv2.calcOpticalFlowFarneback(
    #     frames[0,...], frames[1,...], None,
    #     pyr_scale=0.5, 
    #     levels=5,                            # Lower: to preserve small details
    #     winsize=5,                           # Lower: to catch smaller shifts
    #     iterations=7,                        # Increased: help resolve fast jumps
    #     poly_n=5,
    #     poly_sigma=1.2,
    #     flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN # Upgraded: Better mathematical precision for edges
    # )

    # --- new_flow2 - most sensitive
    # Apply a tiny, sub-pixel blur to smooth out sensor jitter
    # frame1 = cv2.GaussianBlur(frames[0,...], (3, 3), 0.5)       # smoothen to avoid noise
    # frame2 = cv2.GaussianBlur(frames[1,...], (3, 3), 0.5)
    # flow = cv2.calcOpticalFlowFarneback(
    #     frame1, frame2, None,
    #     # frames[0,...], frames[1,...], None,
    #     pyr_scale=0.5,
    #     levels=1,
    #     winsize=3,                                  # or 5
    #     iterations=20,
    #     poly_n=5,
    #     poly_sigma=1.1,
    #     flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
    # )

    u, v = flow[..., 0], flow[..., 1] # Displacement in X and Y direction
    print(type(u))

    # plot magnitude
    magnitude_px = np.sqrt(u**2 + v**2)
    # plt.figure(figsize=(8, 6))
    # plt.imshow(opd, cmap='viridis')
    # plt.colorbar(label='Displacement magnitude [px]')
    # plt.title('Optical Flow Magnitude')
    # plt.show()

    print("Maximum displacement in the sensor [px]:", np.max(magnitude_px))

    phase = reconstruct_from_gradient(u, v)


    # conversion to physical units
    if blender_used:
        config_path = "config_stereo.json"
        with open(config_path, "r") as f:
            config = json.load(f)
        f_mm = config["camera"]["focal_length"]
        sensor_mm = config["camera"]["sensor_size"]
        W_px = config["camera"]["resolution_x"]
        z_A = config["distortions"]["turbulence_distance"][0]
        z_B = config["BOS"]["distance_camera_screen"]
        z_D = z_B - z_A

        f_m  = f_mm * 1e-3                                  # focal length in metres
        f_px = f_mm * (W_px / sensor_mm)                    # focal length in pixels

    else:
        # Configuration of the experimental setup
        B = 0.06        # 6 cm
        # f_mm = 40             # camera specification
        # sensor_mm = 3.84        # camera specification
        # W_px = 1280     # width resolution
        # z_A = 1.04
        # z_B = 1.50       # 0.6 usually
        # z_D = z_B - z_A

        # hamamatsu orcacamera
        f_mm = 40
        sensor_mm = 13.312
        z_A = 0.5
        z_B = 1.45
        z_D = z_B - z_A
        W_px = 2048

        f_m = f_mm * 1e-3                                # focal length in metres
        f_px = (f_mm / sensor_mm) * W_px                # focal length in pixels

    
    S_px = f_px * z_D / (z_D + z_A - f_m)               # [px/rad]
    psi_screen = z_A / f_px                             # [m/px] footprint of a single pixel in the field of view at z_A
    pitch = (sensor_mm / W_px) * 1e-3

    S_m = f_m * z_D / (z_D + z_A - f_m)                 # Sensitivity

    # displacement in the sensor [m]
    delta_x = u * pitch
    delta_y = v * pitch
    displacement_mag = np.sqrt(u**2 + v**2) * pitch
    print("Maximum displacement in the sensor [m]:", np.max(displacement_mag))
    fig = plot_flow_vectors(u, v, step=50, background=frames[0,...])
    plt.savefig(f'flow_vectors{k}.png', dpi=300, bbox_inches='tight')

    # deflection: pixels -> radians
    eps_x = delta_x / S_m
    eps_y = delta_y / S_m
    # deflection_mag = np.sqrt(delta_x**2 + delta_y**2) / S_m
    deflection_mag = np.sqrt(eps_x**2 + eps_y**2)
    print("Maximum deflection [rad]:", np.max(deflection_mag))
    print("Sensitivity [m/rad]:", S_m)
    print("psi screen [m/px]:", psi_screen)

    # displacement in the BOS pattern [m]
    delta_x_background = z_D * np.tan(eps_x)
    delta_y_background = z_D * np.tan(eps_y)
    background_disp_mag = np.sqrt(delta_x_background**2 + delta_y_background**2)


    # phase: pixel-integrated -> meters, technically the optical path difference (OPD)
    opd = reconstruct_from_gradient(eps_x,eps_y) * psi_screen
    print("Minimum OPD [m]:", np.min(opd))

    # temperature prediction
    temperature_field = opd_to_temperature(opd)
    temp_labels = np.linspace(temperature_field.min(), temperature_field.max(), 6)
    print(temp_labels - 273.15)

    # phase: estimate using arbitrary wavelength [rad]
    lambda_0 = 532e-9
    phase1 = (2*np.pi / lambda_0) * opd

    np.save(os.path.join(output_folder,'cam'+str(k)+'_phase_radpx.npy'),phase)
    np.save(os.path.join(output_folder,'cam'+str(k)+'_opd_m.npy'),opd)
    np.save(os.path.join(output_folder,'cam'+str(k)+'_xdeflection_rad.npy'),eps_x)
    np.save(os.path.join(output_folder,'cam'+str(k)+'_ydeflection_rad.npy'),eps_y)
    np.save(os.path.join(output_folder,'cam'+str(k)+'_deflection_mag_rad.npy'),deflection_mag)
    np.save(os.path.join(output_folder,'cam'+str(k)+'_sensor_xdisp_px.npy'),u)
    np.save(os.path.join(output_folder,'cam'+str(k)+'_sensor_ydisp_px.npy'),v)
    np.save(os.path.join(output_folder,'cam'+str(k)+'_sensor_xdisp_m.npy'),delta_x)
    np.save(os.path.join(output_folder,'cam'+str(k)+'_sensor_ydisp_m.npy'),delta_y)
    np.save(os.path.join(output_folder,'cam'+str(k)+'_sensor_disp_mag_m.npy'),displacement_mag)
    np.save(os.path.join(output_folder,'cam'+str(k)+'_background_xdisp_m.npy'),delta_x_background)
    np.save(os.path.join(output_folder,'cam'+str(k)+'_background_ydisp_m.npy'),delta_y_background)
    np.save(os.path.join(output_folder,'cam'+str(k)+'_background_disp_mag_m.npy'),background_disp_mag)
    np.save(os.path.join(output_folder,'cam'+str(k)+'_temperature_field_Kelvin.npy'),temperature_field)


    # --- orcacamera simulation matching ---

    # h = opd / (0.99994 - 1)

    # print("h min/max (metres):", h.min(), h.max())

    # output_file = "data/hamamatsu_phase_nfield.tiff"
    # tiff.imwrite(output_file, n.astype(np.float32))