from pyDCAM import *
from camera import OrcaCamera
from pylablib.devices import DCAM



exposure_time = 0.001
roi =  [0, 1000, 0, 1000]
binning = 1
n_frames_buffer = 1
idx = 0
serial = '003091'
verbose=True

cam = OrcaCamera(exposure_time, roi, binning, n_frames_buffer, idx, serial, verbose)
