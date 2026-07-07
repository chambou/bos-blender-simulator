from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# im = Image.open("data/turbulence_screen6.tiff")
# im = Image.open("data/test_images/wide.tif")
# img = plt.imread("data/test_images/50ms/narrow_dark.tif")
# img = plt.imread("data/test_images/narrow.tif")
# img = plt.imread("data/te.tif")

# plt.figure()
# plt.imshow(np.array(img))
# # plt.imshow(np.array(im),cmap='viridis')
# plt.colorbar()
# plt.show()


# array = np.load('outputs/processing_results/cam0_phase.npy')

# plt.figure()
# plt.imshow(array)
# plt.colorbar()
# plt.show(block=False)

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

# # array = np.load('outputs/processing_results/cam1_phase.npy')

# # plt.figure()
# # plt.imshow(array)
# # plt.colorbar()
# # plt.show(block=False)

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

# L_6 = np.load('../experimental/06072026/sims/with_thickness/separate_screens/6/cam0_phase.npy')
# R_6 = np.load('../experimental/06072026/sims/with_thickness/separate_screens/6/cam1_phase.npy')
# L_7 = np.load('../experimental/06072026/sims/with_thickness/separate_screens/7/cam0_phase.npy')
# R_7 = np.load('../experimental/06072026/sims/with_thickness/separate_screens/7/cam1_phase.npy')
# print(L_6.shape)

# L_sum = L_6 + L_7
# R_sum = R_6 + R_7
# print(L_sum.shape)

# fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6))
# im0 = ax0.imshow(L_6, cmap='viridis')
# ax0.set_title('Left image 6')
# ax0.set_xlim(0, L_sum.shape[1]); ax0.set_ylim(L_sum.shape[0], 0)
# cbar0 = plt.colorbar(im0, ax=ax0)
# cbar0.set_label('Intensity (m)', rotation=270, labelpad=15)

# im1 = ax1.imshow(R_6, cmap='viridis')
# ax1.set_title('Right image 6')
# ax1.set_xlim(0, R_sum.shape[1]); ax1.set_ylim(R_sum.shape[0], 0)
# cbar0 = plt.colorbar(im1, ax=ax1)
# cbar0.set_label('Intensity (m)', rotation=270, labelpad=15)
# plt.show(block=False)

# fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6))
# im0 = ax0.imshow(L_7, cmap='viridis')
# ax0.set_title('Left image 7')
# ax0.set_xlim(0, L_sum.shape[1]); ax0.set_ylim(L_sum.shape[0], 0)
# cbar0 = plt.colorbar(im0, ax=ax0)
# cbar0.set_label('Intensity (m)', rotation=270, labelpad=15)

# im1 = ax1.imshow(R_7, cmap='viridis')
# ax1.set_title('Right image 7')
# ax1.set_xlim(0, R_sum.shape[1]); ax1.set_ylim(R_sum.shape[0], 0)
# cbar0 = plt.colorbar(im1, ax=ax1)
# cbar0.set_label('Intensity (m)', rotation=270, labelpad=15)
# plt.show(block=False)

# fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6))
# im0 = ax0.imshow(L_sum, cmap='viridis')
# ax0.set_title('Left image')
# ax0.set_xlim(0, L_sum.shape[1]); ax0.set_ylim(L_sum.shape[0], 0)
# cbar0 = plt.colorbar(im0, ax=ax0)
# cbar0.set_label('Intensity (m)', rotation=270, labelpad=15)

# im1 = ax1.imshow(R_sum, cmap='viridis')
# ax1.set_title('Right image')
# ax1.set_xlim(0, R_sum.shape[1]); ax1.set_ylim(R_sum.shape[0], 0)
# cbar0 = plt.colorbar(im1, ax=ax1)
# cbar0.set_label('Intensity (m)', rotation=270, labelpad=15)
# plt.show(block=False)

# print(R_sum[892,536])
# print(R_7[892,536])

# def scale(arr):
#     arr_min = arr.min()
#     arr_max = arr.max()

#     normalized = 2 * (arr - arr_min) / (arr_max - arr_min) - 1
#     return normalized

# L_6_norm = scale(L_6)
# R_6_norm = scale(R_6)
# L_7_norm = scale(L_7)
# R_7_norm = scale(R_7)

# L_sum_norm = L_6_norm + L_7_norm
# R_sum_norm = R_6_norm + R_7_norm


# print(R_sum_norm[892,536])
# print(R_7_norm[892,536])

# fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6))
# im0 = ax0.imshow(L_sum_norm, cmap='viridis')
# ax0.set_title('Left image')
# ax0.set_xlim(0, L_sum.shape[1]); ax0.set_ylim(L_sum.shape[0], 0)
# cbar0 = plt.colorbar(im0, ax=ax0)
# cbar0.set_label('Intensity (m)', rotation=270, labelpad=15)

# im1 = ax1.imshow(R_sum_norm, cmap='viridis')
# ax1.set_title('Right image')
# ax1.set_xlim(0, R_sum.shape[1]); ax1.set_ylim(R_sum.shape[0], 0)
# cbar0 = plt.colorbar(im1, ax=ax1)
# cbar0.set_label('Intensity (m)', rotation=270, labelpad=15)
# plt.show()