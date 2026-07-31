import cv2
from pathlib import Path
import glob, os
import numpy as np
import time

def split_cam(frame, path, mode, counter):
    # print(frame)
    center = frame.shape[1]//2
    # img_L = frame[:,:center,0]
    # img_R = frame[:,center:,0]

    img_L = frame[:,:center]
    img_R = frame[:,center]
    
    # Convert to float32 and normalize
    def convert_to_32(frame):
        return (frame.astype(np.float32) / 255.0)

    img_L = convert_to_32(img_L)
    img_R = convert_to_32(img_R)
    print(img_L.shape)
    print(img_L.dtype)

    if mode == "ref":
        L_name = "exp_ref_0deg_0.tif"
        R_name = "exp_ref_0deg_1.tif"
        print("Taking reference image...")
        cv2.imwrite(os.path.join(path,"stereo_ref.tif"), frame)
    elif mode == "with_phase":
        if multiple_imgs:
            L_name = f"exp_0deg_0_{counter}.tif"
            R_name = f"exp_0deg_1_{counter}.tif"
            print("Taking non-reference image...")
            # cv2.imwrite(os.path.join(path,f"stereo_{counter}.tif"), frame)
        elif not multiple_imgs:
            L_name = f"exp_0deg_0.tif"
            R_name = f"exp_0deg_1.tif"
            print("Taking non-reference image...")
            # cv2.imwrite(os.path.join(path,f"stereo.tif"), frame)

    cv2.imwrite(os.path.join(path,L_name), img_L)
    # cv2.imwrite(os.path.join(path,R_name), img_R)

output_folder = Path("outputs/experiments")

# --- Specify mode ---
setup_mode = "ref"
setup_mode = "with_phase"

multiple_imgs = True

modes = ["ref", "non-ref"]

if setup_mode == 'ref':
    print("Reference image mode...")
else:
    if multiple_imgs:
        print("Turbulence image mode with multiple images...")
    elif not multiple_imgs:
        print("Turbulence image mode with single image...")
        
# --------------------

# Open the default camera
# cam = cv2.VideoCapture(0)

# # Get the default frame width and height
# frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# cv2.namedWindow("test")

# img_counter = 0

# previous_frame = None

# while True:
#     ret, frame = cam.read()
#     if not ret:
#         print("failed to grab frame")
#         break
#     cv2.imshow("test", frame)

#     current_gray = frame[:, :, 0].copy()

#     if previous_frame is not None:
#         diff_signed = current_gray.astype(np.int16) - previous_frame.astype(np.int16)

#         # Convert to a displayable image: centre zero at mid-gray (128),
#         # so both positive and negative differences are visible
#         diff_display = np.clip(diff_signed + 128, 0, 255).astype(np.uint8)

#         cv2.imshow("difference", diff_display)

#     k = cv2.waitKey(1)
#     if k%256 == 27:
#         # ESC pressed
#         print("Escape hit, closing...")
#         break
#     elif k%256 == 32:
#         # SPACE pressed
#         img_name = "test_image_{}.tif".format(img_counter)
#         # cv2.imwrite(os.path.join(output_folder,img_name), frame)
#         split_cam(frame, output_folder, setup_mode, img_counter)
#         if setup_mode == "with_phase":
#             print(f"{img_counter} written!")
#             img_counter += 1
    
#     previous_frame = current_gray

# cam.release()

# cv2.destroyAllWindows()



# --- Video Mode ---

# # --- CONFIGURATION ---
# target_fps = 30  # Set desired frame rate
# output_dir = "captured_frames_tif_4"
# os.makedirs(output_dir, exist_ok=True)

# cap = cv2.VideoCapture(0)

# # Do NOT set CAP_PROP_FRAME_WIDTH / HEIGHT — leaves image at original native size

# frame_delay = 1.0 / target_fps
# captured_frames = []

# print(
#     f"Capturing native resolution (Channel 0) at {target_fps} FPS. Press 'q' to stop."
# )

# # --- FAST CAPTURE LOOP ---
# while True:
#     loop_start = time.time()

#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Extract 1st channel (Index 0 = Blue in OpenCV BGR) at ORIGINAL dimensions
#     ch1_frame = frame[:, :, 0]

#     captured_frames.append(ch1_frame)

#     cv2.imshow("1st Channel Feed", ch1_frame)

#     # Maintain target FPS
#     elapsed = time.time() - loop_start
#     sleep_time = frame_delay - elapsed
#     if sleep_time > 0:
#         time.sleep(sleep_time)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()

# # --- BATCH SAVE UNMODIFIED TIFFs ---
# if captured_frames:
#     height, width = captured_frames[0].shape
#     print(
#         f"Captured {len(captured_frames)} images at native resolution ({width}x{height})."
#     )
#     print(f"Saving to '{output_dir}/'...")

#     for i, frame_ch in enumerate(captured_frames):
#         filename = os.path.join(output_dir, f"frame_{i:04d}.tif")
#         cv2.imwrite(filename, frame_ch)

#     print("Done saving native TIF files!")


# --- To split the images using video mode ---

input_folder = Path('../experimental/31072026/multiple_frames/captured_frames_tif_3')
output_folder = Path('../experimental/31072026/multiple_frames/captured_frames_tif_3/left_images')

ext = ".tif"
# ------------------------------
# LOAD IMAGE LIST
# ------------------------------
files = sorted(glob.glob(os.path.join(input_folder, '*'+ext)))
read_files = [cv2.imread(f, cv2.IMREAD_UNCHANGED) for f in files]
frames = [cv2.normalize(f, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)for f in read_files]
frames[0].shape
# imgs = [split_cam(f, output_folder, setup_mode, img_counter) for f in frames]

img_counter = 0
for k in range(0, len(frames)):
    split_cam(frames[k], output_folder, setup_mode, img_counter)
    img_counter += 1