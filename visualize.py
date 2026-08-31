# Script to visualize evrything (comment and uncomment by choice)

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import os

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


array = np.load('outputs/processing_results/cam0_deflection_mag_rad.npy')
plt.figure(figsize=(8, 6))
plt.imshow(array, cmap='viridis')
tick_locs = np.linspace(array.min(), array.max(), 6)
cbar_L1 = plt.colorbar()
cbar_L1.set_label('Deflection Magnitude [rad]')
cbar_L1.set_ticks(tick_locs)
plt.show(block=False)


array = np.load('outputs/processing_results/cam0_opd_m.npy')

plt.figure(figsize=(8, 6))
plt.imshow(array, cmap='viridis')
tick_locs = np.linspace(array.min(), array.max(), 6)
cbar_L1 = plt.colorbar()
cbar_L1.set_label('Optical Path Difference (OPD) [m]')
cbar_L1.set_ticks(tick_locs)
plt.show(block=False)

# plt.figure()
# L = 0.33599999999999997
# x = np.linspace(-L/2, L/2, 1080)
# X, Y = np.meshgrid(x, x)
# r = np.hypot(X, Y)
# array = array - array.mean()
# simu_opd_normalized = array / np.max(array)
# print(simu_opd_normalized.max())
# # plt.plot(r, opd_normalized)
# # plt.title("Simulation Normalized")
# # plt.show(block=False)
# np.save('outputs/simulation_double_int_opd_normalized.npy',simu_opd_normalized)

# analytic_opd_norm = np.load('outputs/analytical_double_int_opd_normalized.npy')
# residuals = simu_opd_normalized - analytic_opd_norm
# rms_error = np.sqrt(np.mean(residuals**2))
# rms_percent = rms_error / np.abs(analytic_opd_norm).max() * 100

# fig,ax = plt.subplots(figsize=(7, 5))

# ax.plot(r, analytic_opd_norm, 'r-', label='Analytic', linewidth=2)
# ax.scatter(r, simu_opd_normalized, s=1, alpha=0.3, color='C0', label='Simulation')
# ax.set_xlabel('$r$ [m]')
# ax.set_ylabel('Normalized OPD [m]')
# # ax.title('Analytic vs. Simulated OPD, Single-Interface Case')
# ax.set_ylim(analytic_opd_norm.min(), analytic_opd_norm.max())
# ax.set_xlim(r.min(), r.max())
# xticks = np.linspace(r.min(), r.max(), 5)
# xlabels = [f"{num:.2f}" for num in xticks]
# ax.set_xticks(
#     ticks=xticks,
#     labels=xlabels
# )
# handles, labels = plt.gca().get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# ax.legend(by_label.values(), by_label.keys())
# stats_text = f"RMS error: {rms_error:.4e} m ({rms_percent:.2f}%)"
# ax.text(
#     0.05, 0.95, stats_text,
#     transform=ax.transAxes,
#     verticalalignment='top',
#     fontsize=10,
#     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
# )

# plt.tight_layout()
# plt.show(block=False)

# array = np.load('outputs/processing_results/cam0_sensor_xdisp_m.npy')

# plt.figure()
# plt.imshow(array)
# cbar0 = plt.colorbar()
# cbar0.set_label('(m)', rotation=270, labelpad=15)
# plt.show(block=False)

# array = np.load('outputs/processing_results/cam0_sensor_ydisp_m.npy')

# plt.figure()
# plt.imshow(array)
# cbar0 = plt.colorbar()
# cbar0.set_label('(m)', rotation=270, labelpad=15)
# plt.show()

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

# ## ------------- Deflection [rad] -----------------

# array_left = np.load('outputs/processing_results/cam0_deflection_mag_rad.npy')
# array_right = np.load('outputs/processing_results/cam1_deflection_mag_rad.npy')

# fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6))
# fig.suptitle(('Deflection'))
# im0 = ax0.imshow(array_left,vmax=1e-7, cmap='hot')
# ax0.set_title('Left image')
# ax0.set_xlim(0, array_left.shape[1]); ax0.set_ylim(array_left.shape[0], 0)
# cbar0 = plt.colorbar(im0, ax=ax0)
# cbar0.set_label('Magnitude (rad)', rotation=270, labelpad=15)

# im1 = ax1.imshow(array_right,vmax=1e-7, cmap='hot')
# ax1.set_title('Right image')
# ax1.set_xlim(0, array_right.shape[1]); ax1.set_ylim(array_right.shape[0], 0)
# cbar0 = plt.colorbar(im1, ax=ax1)
# cbar0.set_label('Magnitude (rad)', rotation=270, labelpad=15)
# plt.tight_layout()
# plt.show(block=False)

# ## ------------- Sensor displacement [m] -----------------

# array_left = np.load('outputs/processing_results/cam0_sensor_disp_mag_m.npy')
# array_right = np.load('outputs/processing_results/cam1_sensor_disp_mag_m.npy')
# print(type(array_left))

# fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6))
# fig.suptitle(('Displacement in the Sensor'))
# im0 = ax0.imshow(array_left,vmax=3e-8, cmap='hot')
# ax0.set_title('Left image')
# ax0.set_xlim(0, array_left.shape[1]); ax0.set_ylim(array_left.shape[0], 0)
# cbar0 = plt.colorbar(im0, ax=ax0)
# cbar0.set_label('Magnitude (m)', rotation=270, labelpad=15)

# im1 = ax1.imshow(array_right,vmax=3e-8, cmap='hot')
# ax1.set_title('Right image')
# ax1.set_xlim(0, array_right.shape[1]); ax1.set_ylim(array_right.shape[0], 0)
# cbar0 = plt.colorbar(im1, ax=ax1)
# cbar0.set_label('Magnitude (m)', rotation=270, labelpad=15)
# plt.tight_layout()
# plt.show(block=False)

# ## ------------- Background displacement [m] -----------------

# array_left = np.load('outputs/processing_results/cam0_background_disp_mag_m.npy')
# array_right = np.load('outputs/processing_results/cam1_background_disp_mag_m.npy')

# fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6))
# fig.suptitle(('Displacement in the Background'))
# im0 = ax0.imshow(array_left,vmax=1e-7, cmap='hot')
# ax0.set_title('Left image')
# ax0.set_xlim(0, array_left.shape[1]); ax0.set_ylim(array_left.shape[0], 0)
# cbar0 = plt.colorbar(im0, ax=ax0)
# cbar0.set_label('Magnitude (m)', rotation=270, labelpad=15)

# im1 = ax1.imshow(array_right,vmax=1e-7, cmap='hot')
# ax1.set_title('Right image')
# ax1.set_xlim(0, array_right.shape[1]); ax1.set_ylim(array_right.shape[0], 0)
# cbar0 = plt.colorbar(im1, ax=ax1)
# cbar0.set_label('Magnitude (m)', rotation=270, labelpad=15)
# plt.tight_layout()
# plt.show()


# # array0x = np.load('outputs/processing_results/cam0_xdisp_m.npy')
# # array0y = np.load('outputs/processing_results/cam0_ydisp_m.npy')

# # array1x = np.load('outputs/processing_results/cam1_xdisp_m.npy')
# # array1y = np.load('outputs/processing_results/cam1_ydisp_m.npy')

# # magnitude0 = np.sqrt(array0x**2 + array0y**2)
# # magnitude1 = np.sqrt(array1x**2 + array1y**2)

# # plt.figure(figsize=(8, 6))
# # # plt.imshow(magnitude, cmap='hot')
# # plt.imshow(magnitude0,vmax=0.5, cmap='hot')
# # plt.colorbar(label='magnitude')
# # plt.title('Optical Flow Magnitude')
# # plt.show(block=False)

# # plt.figure(figsize=(8, 6))
# # # plt.imshow(magnitude1, cmap='hot')
# # plt.imshow(magnitude1,vmax=1e-8, cmap='hot')
# # plt.colorbar(label='magnitude')
# # plt.title('Optical Flow Magnitude')
# # plt.show()

num = 0

# array_left = np.load(f'../experimental/27072026/stereo_40mm/sample_2/data/cam{num}_sensor_disp_mag_m.npy')
# array_right = np.load(f'../experimental/27072026/stereo_40mm/sample_2/data/cam{num}_deflection_mag_rad.npy')
array_left = np.load(f'outputs/processing_results/cam{num}_sensor_disp_mag_m.npy')
array_right = np.load(f'outputs/processing_results/cam{num}_deflection_mag_rad.npy')
# print(type(array_left))
# print(array_left.max() * 1e6)
# print(array_right.max() * 1e3)
# print(array.min())

# fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6))
# im0 = ax0.imshow(array_left, cmap='hot')
# ax0.set_xlim(0, array_left.shape[1]); ax0.set_ylim(array_left.shape[0], 0)
# cbar0 = plt.colorbar(im0, ax=ax0)
# cbar0.set_label('Displacement in the Sensor [m]')

# im1 = ax1.imshow(array_right*1e3, cmap='hot')
# ax1.set_xlim(0, array_right.shape[1]); ax1.set_ylim(array_right.shape[0], 0)
# cbar0 = plt.colorbar(im1, ax=ax1)
# cbar0.set_label('Angular Deflection [mrad]')
# plt.tight_layout()
# plt.show()


# display displacement in the sensor and deflection magnitude in one plot

fig = plt.figure(figsize=(9, 6))
gs = gridspec.GridSpec(1, 3, width_ratios=[20, 1, 1], wspace=0.6)

ax = fig.add_subplot(gs[0])
cax1 = fig.add_subplot(gs[1])
cax2 = fig.add_subplot(gs[2])

# Explicit vmin/vmax - removes any ambiguity about what range the colormap spans
vmin, vmax = array_left.min(), array_left.max()
im = ax.imshow(array_left, cmap='hot', vmin=vmin, vmax=vmax)

# Explicit tick positions, matching vmin/vmax exactly
tick_locs = np.linspace(vmin, vmax, 6)

cbar1 = fig.colorbar(im, cax=cax1)
cbar1.set_ticks(tick_locs)
cbar1.set_label('Displacement in the Sensor [m]')

cbar2 = fig.colorbar(im, cax=cax2)
cbar2.set_ticks(tick_locs)   # SAME positions on the shared color scale
S_m = 0.012602739726027398
mrad_labels = [f"{(t/S_m)*1000:.2f}" for t in tick_locs]
# mrad_labels = []
# for t in tick_locs:
#     idx = np.unravel_index(np.argmin(np.abs(opd_array - t)), opd_array.shape)
#     temp_labels.append(f"{temp_array[idx] - 273.15:.1f}")

cbar2.set_ticklabels(mrad_labels)
cbar2.set_label('Angular Deflection [mrad]')


plt.show(block=False)

# display opd and temperature in one plot
opd_array = np.load('outputs/processing_results/cam0_opd_m.npy')
temp_array = np.load('outputs/processing_results/cam0_temperature_field_Kelvin.npy')

fig = plt.figure(figsize=(9, 6))
gs = gridspec.GridSpec(1, 3, width_ratios=[20, 1, 1], wspace=0.6)

ax = fig.add_subplot(gs[0])
cax1 = fig.add_subplot(gs[1])
cax2 = fig.add_subplot(gs[2])

im = ax.imshow(opd_array, cmap='viridis')

cbar1 = fig.colorbar(im, cax=cax1)
cbar1.set_label('Optical Path Difference [m]')

cbar2 = fig.colorbar(im, cax=cax2)
cbar2.set_label('Temperature [°C]')

# Tick positions on the shared color scale (from OPD data)
tick_locs = np.linspace(opd_array.min(), opd_array.max(), 6)
cbar1.set_ticks(tick_locs)
cbar2.set_ticks(tick_locs)

# For each OPD tick position, find the CORRESPONDING temperature value 
# by looking it up directly from your existing temperature array, 
# at the pixel(s) where OPD is closest to that tick value
temp_labels = []
for t in tick_locs:
    idx = np.unravel_index(np.argmin(np.abs(opd_array - t)), opd_array.shape)
    temp_labels.append(f"{temp_array[idx] - 273.15:.1f}")

cbar2.set_ticklabels(temp_labels)

plt.show(block=False)


#--- deflection
# array_left = np.load(f'outputs/processing_results/cam{num}_xdeflection_rad.npy')
# array_right = np.load(f'outputs/processing_results/cam{num}_ydeflection_rad.npy')

# fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6))
# im0 = ax0.imshow(array_left, cmap='viridis')
# ax0.set_title('x-axis deflection')
# ax0.set_xlim(0, array_left.shape[1]); ax0.set_ylim(array_left.shape[0], 0)
# cbar0 = plt.colorbar(im0, ax=ax0)
# cbar0.set_label('(rad)', rotation=270, labelpad=15)

# im1 = ax1.imshow(array_right, cmap='viridis')
# ax1.set_title('y-axis deflection')
# ax1.set_xlim(0, array_right.shape[1]); ax1.set_ylim(array_right.shape[0], 0)
# cbar0 = plt.colorbar(im1, ax=ax1)
# cbar0.set_label('(rad)', rotation=270, labelpad=15)

# plt.tight_layout()
# plt.show()


num = 0

# array_left = np.load(f'../experimental/27072026/stereo_40mm/sample_2/data/cam{num}_sensor_disp_mag_m.npy')
# array_right = np.load(f'../experimental/27072026/stereo_40mm/sample_2/data/cam{num}_deflection_mag_rad.npy')
disp_left = np.load(f'outputs/processing_results/cam0_sensor_disp_mag_m.npy')
disp_right = np.load(f'outputs/processing_results/cam1_sensor_disp_mag_m.npy')
def_left = np.load(f'outputs/processing_results/cam0_deflection_mag_rad.npy')
def_right = np.load(f'outputs/processing_results/cam1_deflection_mag_rad.npy')


# display displacement in the sensor and deflection magnitude in one plot for left and right camera

# --- Figure layout: image, cbar, cbar | image, cbar, cbar ---
fig = plt.figure(figsize=(16, 6))
gs = gridspec.GridSpec(1, 9, width_ratios=[20, 1, 0.5, 1, 1, 20, 1, 0.5, 1], wspace=0.5)

ax_L   = fig.add_subplot(gs[0])
cax_L1 = fig.add_subplot(gs[1])
cax_L2 = fig.add_subplot(gs[3])
ax_R   = fig.add_subplot(gs[5])
cax_R1 = fig.add_subplot(gs[6])
cax_R2 = fig.add_subplot(gs[8])


# --- Left camera ---
im_L = ax_L.imshow(disp_left, cmap='hot')
ax_L.set_title('Left Camera')

tick_locs = np.linspace(disp_left.min(), disp_left.max(), 6)

cbar_L1 = fig.colorbar(im_L, cax=cax_L1)
cbar_L1.set_label('Displacement in the Sensor [m]')
cbar_L1.set_ticks(tick_locs)

cbar_L2 = fig.colorbar(im_L, cax=cax_L2)
cbar_L2.set_ticks(tick_locs)
S_m = 0.00404040404040404      # sensitivity
mrad_labels = [f"{(t/S_m)*1000:.2f}" for t in tick_locs]
cbar_L2.set_ticklabels(mrad_labels)
cbar_L2.set_label('Deflection Angle [mrad]')

# --- Right camera ---
im_R = ax_R.imshow(disp_right, cmap='hot')
ax_R.set_title('Right Camera')

tick_locs = np.linspace(disp_right.min(), disp_right.max(), 6)

cbar_R1 = fig.colorbar(im_R, cax=cax_R1)
cbar_R1.set_label('Displacement in the Sensor [m]')
cbar_R1.set_ticks(tick_locs)

cbar_R2 = fig.colorbar(im_R, cax=cax_R2)
cbar_R2.set_ticks(tick_locs)
S_m = 0.00404040404040404      # sensitivity
mrad_labels = [f"{(t/S_m)*1000:.2f}" for t in tick_locs]
cbar_R2.set_ticklabels(mrad_labels)
cbar_R2.set_label('Deflection Angle [mrad]')

plt.show(block=False)

#---------opd and temperatude for left and right images

opd_left = np.load('outputs/processing_results/cam0_opd_m.npy')
opd_right = np.load('outputs/processing_results/cam1_opd_m.npy')
temp_left = np.load('outputs/processing_results/cam0_temperature_field_Kelvin.npy')
temp_right = np.load('outputs/processing_results/cam1_temperature_field_Kelvin.npy')


# display displacement in the sensor and deflection magnitude in one plot for left and right camera

# --- Figure layout: image, cbar, cbar | image, cbar, cbar ---
fig = plt.figure(figsize=(16, 6))
gs = gridspec.GridSpec(1, 9, width_ratios=[20, 1, 0.5, 1, 1, 20, 1, 0.5, 1], wspace=0.5)

ax_L   = fig.add_subplot(gs[0])
cax_L1 = fig.add_subplot(gs[1])
cax_L2 = fig.add_subplot(gs[3])
ax_R   = fig.add_subplot(gs[5])
cax_R1 = fig.add_subplot(gs[6])
cax_R2 = fig.add_subplot(gs[8])


# --- Left camera ---
im_L = ax_L.imshow(opd_left, cmap='viridis')
ax_L.set_title('Left Camera')

tick_locs = np.linspace(opd_left.min(), opd_left.max(), 6)

cbar_L1 = fig.colorbar(im_L, cax=cax_L1)
cbar_L1.set_label('Optical Path Difference [m]')
cbar_L1.set_ticks(tick_locs)

cbar_L2 = fig.colorbar(im_L, cax=cax_L2)
cbar_L2.set_ticks(tick_locs)
temp_labels = []
for t in tick_locs:
    idx = np.unravel_index(np.argmin(np.abs(opd_left - t)), opd_left.shape)
    temp_labels.append(f"{temp_left[idx] - 273.15:.1f}")
cbar_L2.set_ticklabels(temp_labels)
cbar_L2.set_label('Temperature [°C]')

# --- Right camera ---
im_R = ax_R.imshow(opd_right, cmap='viridis')
ax_R.set_title('Right Camera')

tick_locs = np.linspace(opd_right.min(), opd_right.max(), 6)

cbar_R1 = fig.colorbar(im_R, cax=cax_R1)
cbar_R1.set_label('Optical Path Difference [m]')
cbar_R1.set_ticks(tick_locs)

cbar_R2 = fig.colorbar(im_R, cax=cax_R2)
cbar_R2.set_ticks(tick_locs)
temp_labels = []
for t in tick_locs:
    idx = np.unravel_index(np.argmin(np.abs(opd_right - t)), opd_right.shape)
    temp_labels.append(f"{temp_right[idx] - 273.15:.1f}")
cbar_R2.set_ticklabels(temp_labels)
cbar_R2.set_label('Temperature [°C]')

plt.show()