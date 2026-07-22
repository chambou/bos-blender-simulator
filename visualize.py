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

# array = np.load('outputs/processing_results/cam0_xdisp.npy')

# plt.figure()
# plt.imshow(array)
# cbar0 = plt.colorbar()
# cbar0.set_label('(rad)', rotation=270, labelpad=15)
# plt.show(block=False)

# array = np.load('outputs/processing_results/cam0_ydisp.npy')

# plt.figure()
# plt.imshow(array)
# cbar0 = plt.colorbar()
# cbar0.set_label('(rad)', rotation=270, labelpad=15)
# plt.show(block=False)

# # array = np.load('outputs/processing_results/cam1_phase.npy')

# # plt.figure()
# # plt.imshow(array)
# # plt.colorbar()
# # plt.show(block=False)

# array = np.load('outputs/processing_results/cam1_xdisp.npy')

# plt.figure()
# plt.imshow(array)
# cbar0 = plt.colorbar()
# cbar0.set_label('(rad)', rotation=270, labelpad=15)
# plt.show(block=False)

# array = np.load('outputs/processing_results/cam1_ydisp.npy')

# plt.figure()
# plt.imshow(array)
# cbar0 = plt.colorbar()
# cbar0.set_label('(rad)', rotation=270, labelpad=15)
# plt.show()

# array = np.load('outputs/processing_results/glass_prediction/analytical_phase.npy')

# plt.figure()
# plt.imshow(array)
# cbar0 = plt.colorbar()
# cbar0.set_label('(m)', rotation=270, labelpad=15)
# plt.show(block=False)

# array = np.load('outputs/processing_results/glass_prediction/analytical_xdisp.npy')

# plt.figure()
# plt.imshow(array)
# cbar0 = plt.colorbar()
# cbar0.set_label('(rad)', rotation=270, labelpad=15)
# plt.show(block=False)

# array = np.load('outputs/processing_results/glass_prediction/analytical_ydisp.npy')

# plt.figure()
# plt.imshow(array)
# cbar0 = plt.colorbar()
# cbar0.set_label('(rad)', rotation=270, labelpad=15)
# plt.show()

# L_6 = np.load('../experimental/10072026/gaussian/nonsolidified/separate_screens/near/cam0_phase.npy')
# R_6 = np.load('../experimental/10072026/gaussian/nonsolidified/separate_screens/near/cam1_phase.npy')
# L_7 = np.load('../experimental/10072026/gaussian/nonsolidified/separate_screens/far/cam0_phase.npy')
# R_7 = np.load('../experimental/10072026/gaussian/nonsolidified/separate_screens/far/cam1_phase.npy')
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

## ------------- Sensor displacement [m] -----------------

array_left = np.load('outputs/processing_results/cam0_sensor_disp_mag_m.npy')
array_right = np.load('outputs/processing_results/cam1_sensor_disp_mag_m.npy')

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(('Sensor Displacement'))
im0 = ax0.imshow(array_left,vmax=5e-8, cmap='hot')
ax0.set_title('Left image')
ax0.set_xlim(0, array_left.shape[1]); ax0.set_ylim(array_left.shape[0], 0)
cbar0 = plt.colorbar(im0, ax=ax0)
cbar0.set_label('Magnitude (m)', rotation=270, labelpad=15)

im1 = ax1.imshow(array_right,vmax=5e-8, cmap='hot')
ax1.set_title('Right image')
ax1.set_xlim(0, array_right.shape[1]); ax1.set_ylim(array_right.shape[0], 0)
cbar0 = plt.colorbar(im1, ax=ax1)
cbar0.set_label('Magnitude (m)', rotation=270, labelpad=15)
plt.tight_layout()
plt.show()

# array0x = np.load('outputs/processing_results/cam0_xdisp_m.npy')
# array0y = np.load('outputs/processing_results/cam0_ydisp_m.npy')

# array1x = np.load('outputs/processing_results/cam1_xdisp_m.npy')
# array1y = np.load('outputs/processing_results/cam1_ydisp_m.npy')

# magnitude0 = np.sqrt(array0x**2 + array0y**2)
# magnitude1 = np.sqrt(array1x**2 + array1y**2)

# plt.figure(figsize=(8, 6))
# # plt.imshow(magnitude, cmap='hot')
# plt.imshow(magnitude0,vmax=0.5, cmap='hot')
# plt.colorbar(label='magnitude')
# plt.title('Optical Flow Magnitude')
# plt.show(block=False)

# plt.figure(figsize=(8, 6))
# # plt.imshow(magnitude1, cmap='hot')
# plt.imshow(magnitude1,vmax=1e-8, cmap='hot')
# plt.colorbar(label='magnitude')
# plt.title('Optical Flow Magnitude')
# plt.show()