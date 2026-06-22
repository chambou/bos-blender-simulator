from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

im = Image.open("data/turbulence_screen6.tiff")

plt.figure()
plt.imshow(np.array(im),cmap='viridis')
plt.colorbar()
plt.show(block=False)


array = np.load('outputs/processing_results/cam0_phase.npy')

plt.figure()
plt.imshow(array)
plt.colorbar()
plt.show(block=False)

array = np.load('outputs/processing_results/cam0_xdisp.npy')

plt.figure()
plt.imshow(array)
plt.colorbar()
plt.show(block=False)

array = np.load('outputs/processing_results/cam0_ydisp.npy')

plt.figure()
plt.imshow(array)
plt.colorbar()
plt.show(block=False)

array = np.load('outputs/processing_results/cam1_phase.npy')

plt.figure()
plt.imshow(array)
plt.colorbar()
plt.show(block=False)

array = np.load('outputs/processing_results/cam1_xdisp.npy')

plt.figure()
plt.imshow(array)
plt.colorbar()
plt.show(block=False)

array = np.load('outputs/processing_results/cam1_ydisp.npy')

plt.figure()
plt.imshow(array)
plt.colorbar()
plt.show()

# array = np.load('outputs/processing_results/cam0_phase_pred.npy')

# plt.figure()
# plt.imshow(array)
# plt.colorbar()
# plt.show(block=False)

# array = np.load('outputs/processing_results/cam0_xdisp_pred.npy')

# plt.figure()
# plt.imshow(array)
# plt.colorbar()
# plt.show(block=False)

# array = np.load('outputs/processing_results/cam0_ydisp_pred.npy')

# plt.figure()
# plt.imshow(array)
# plt.colorbar()
# plt.show()
