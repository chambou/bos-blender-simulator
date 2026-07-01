from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# im = Image.open("data/turbulence_screen6.tiff")
im = Image.open("data/test_images/wide.tif")
img = plt.imread("data/test_images/50ms/narrow_dark.tif")
img = plt.imread("data/test_images/narrow.tif")

plt.figure()
plt.imshow(np.array(img))
# plt.imshow(np.array(im),cmap='viridis')
plt.colorbar()
plt.show()


# array = np.load('outputs/processing_results/cam0_phase.npy')

# plt.figure()
# plt.imshow(array)
# plt.colorbar()
# plt.show(block=False)

# array = np.load('outputs/processing_results/cam0_xdisp.npy')

# plt.figure()
# plt.imshow(array)
# plt.colorbar()
# plt.show(block=False)

# array = np.load('outputs/processing_results/cam0_ydisp.npy')

# plt.figure()
# plt.imshow(array)
# plt.colorbar()
# plt.show(block=False)

# array = np.load('outputs/processing_results/cam1_phase.npy')

# plt.figure()
# plt.imshow(array)
# plt.colorbar()
# plt.show(block=False)

# array = np.load('outputs/processing_results/cam1_xdisp.npy')

# plt.figure()
# plt.imshow(array)
# plt.colorbar()
# plt.show(block=False)

# array = np.load('outputs/processing_results/cam1_ydisp.npy')

# plt.figure()
# plt.imshow(array)
# plt.colorbar()
# plt.show()

# # array = np.load('outputs/processing_results/cam0_phase_pred.npy')

# # plt.figure()
# # plt.imshow(array)
# # plt.colorbar()
# # plt.show(block=False)

# # array = np.load('outputs/processing_results/cam0_xdisp_pred.npy')

# # plt.figure()
# # plt.imshow(array)
# # plt.colorbar()
# # plt.show(block=False)

# # array = np.load('outputs/processing_results/cam0_ydisp_pred.npy')

# # plt.figure()
# # plt.imshow(array)
# # plt.colorbar()
# # plt.show()
