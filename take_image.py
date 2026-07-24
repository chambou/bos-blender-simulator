# import cv2
# from pathlib import Path
# import glob, os
# import numpy as np
# import time

# def split_cam(frame, path, mode):
#     center = frame.shape[1]//2
#     img_L = frame[:,:center,0]
#     img_R = frame[:,center:,0]
    
#     # Convert to float32 and normalize
#     def convert_to_32(frame):
#         return (frame.astype(np.float32) / 255.0)

#     img_L = convert_to_32(img_L)
#     img_R = convert_to_32(img_R)
#     print(img_L.shape)
#     print(img_L.dtype)

#     if mode == "ref":
#         L_name = "exp_ref_0deg_0.tif"
#         R_name = "exp_ref_0deg_1.tif"
#         print("Taking reference image...")
#         cv2.imwrite(os.path.join(path,"stereo_ref.tif"), frame)
#     elif mode == "with_phase":
#         L_name = "exp_0deg_0.tif"
#         R_name = "exp_0deg_1.tif"
#         print("Taking non-reference image...")
#         cv2.imwrite(os.path.join(path,"stereo.tif"), frame)

#     cv2.imwrite(os.path.join(path,L_name), img_L)
#     cv2.imwrite(os.path.join(path,R_name), img_R)

# output_folder = Path("outputs/experiments")

# # --- Specify mode ---
# setup_mode = "ref"
# setup_mode = "with_phase"

# modes = ["ref", "non-ref"]

# if setup_mode == 'ref':
#     print("Reference image mode...")
# else:

#     print("Turbulence image mode...")
# # --------------------

# # Open the default camera
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
#         split_cam(frame, output_folder, setup_mode)
#         # print("{} written!".format(img_name))
#         img_counter += 1
    
#     previous_frame = current_gray

# cam.release()

# cv2.destroyAllWindows()



# 

# import pymmcore
# import cv2
# import numpy as np

# # Initialize Micro-Manager Core
# mmc = pymmcore.CMMCore()

# # Load the macOS-compatible Hamamatsu device adapter
# mmc.loadDevice("Camera", "HamamatsuHam", "HamamatsuHam_Cam")
# mmc.initializeDevice("Camera")

# # Configure exposure time (in milliseconds)
# mmc.setExposure(20.0)

# # Capture a frame
# mmc.snapImage()
# raw_img = mmc.getImage() # Returns raw uint16 array

# # Convert to float32 for Optical Flow / BOS processing
# float_img = raw_img.astype(np.float32) / 65535.0

# print(f"Captured frame shape: {float_img.shape}, dtype: {float_img.dtype}")

import os
import pymmcore
import cv2
import numpy as np

# 1. Instantiate the MMCore controller
mmc = pymmcore.CMMCore()

# 2. Point pymmcore to your Micro-Manager installation path on macOS
# (Adjust the path if your version folder name is slightly different)
mm_path = "/Applications/Micro-Manager-2.0"
mmc.setDeviceAdapterSearchPaths([mm_path])

# 3. Load the Hamamatsu device adapter
# Note: On macOS, use "Hamamatsu" (legacy adapter) instead of "HamamatsuHam"
try:
    mmc.loadDevice("Camera", "Hamamatsu", "Hamamatsu-DCAM")
except Exception as e:
    # Fallback to HamamatsuHam if using a newer custom compiled adapter
    mmc.loadDevice("Camera", "HamamatsuHam", "HamamatsuHam_Cam")

# 4. Initialize the camera
mmc.initializeDevice("Camera")
mmc.setCameraDevice("Camera")

# 5. Configure exposure (in milliseconds)
mmc.setExposure(20.0)

print("Camera initialized successfully on macOS!")

# 6. Capture a test frame
mmc.snapImage()
raw_array = mmc.getImage() # Returns uint16 numpy array

# Convert to float32 [0.0, 1.0] for optical flow / BOS
float_array = raw_array.astype(np.float32) / 65535.0

print(f"Captured frame shape: {float_array.shape}, dtype: {float_array.dtype}")