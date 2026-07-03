import cv2
from pathlib import Path
import glob, os

# # Open the default camera
# cam = cv2.VideoCapture(0)

# # Get the default frame width and height
# frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

def split_cam(frame, path, mode):
    center = frame.shape[1]//2
    img_L = frame[:,:center,0]
    img_R = frame[:,center:,0]

    if mode == "ref":
        L_name = "exp_ref_0deg_0.tif"
        R_name = "exp_ref_0deg_1.tif"
        print("Taking reference image...")
        cv2.imwrite(os.path.join(path,"stereo_ref.tif"), frame)
    elif mode == "with_phase":
        L_name = "exp_0deg_0.tif"
        R_name = "exp_0deg_1.tif"
        print("Taking non-reference image...")
        cv2.imwrite(os.path.join(path,"stereo.tif"), frame)

    cv2.imwrite(os.path.join(path,L_name), img_L)
    cv2.imwrite(os.path.join(path,R_name), img_R)

output_folder = Path("outputs/experiments")

# --- Specify mode ---
setup_mode = "ref"
setup_mode = "with_phase"
# --------------------

# Open the default camera
cam = cv2.VideoCapture(0)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

cv2.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "test_image_{}.tif".format(img_counter)
        # cv2.imwrite(os.path.join(output_folder,img_name), frame)
        split_cam(frame, output_folder, setup_mode)
        # print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()

