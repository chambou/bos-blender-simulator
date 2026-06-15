from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

im = Image.open("data/turbulence_screen.tiff")

plt.figure()
plt.imshow(np.array(im),cmap='viridis')
plt.show(block=False)


array = np.load('outputs/processing_results/cam0_phase.npy')

plt.figure()
plt.imshow(array)
plt.show(block=False)

array = np.load('outputs/processing_results/cam0_xdisp.npy')

plt.figure()
plt.imshow(array)
plt.show(block=False)
plt.colorbar()

array = np.load('outputs/processing_results/cam0_ydisp.npy')

plt.figure()
plt.imshow(array)
plt.show()
plt.colorbar()