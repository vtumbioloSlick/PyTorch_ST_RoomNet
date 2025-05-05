import scipy.io
import matplotlib.pyplot as plt

mat = scipy.io.loadmat("sun_dumwbpyrlvlpkrxs.mat")
layout = mat['layout']

plt.imshow(layout, cmap='gray')
plt.title("Layout from .mat")
plt.colorbar()
plt.show()
