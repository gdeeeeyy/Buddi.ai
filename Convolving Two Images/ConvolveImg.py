import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.signal as sp

img=Image.open("Convolving Two Images/img.JPG")
imgBw=img.convert("1")
plt.imshow(imgBw)
plt.show()

arrayImg=np.array(imgBw, dtype=np.float64)
print(arrayImg.shape)

kerImg=Image.open("Convolving Two Images/kernel.png")
kerBw=kerImg.convert("1")
plt.imshow(kerBw)
plt.show()

arrayKer=np.array(kerBw, dtype=np.float64)
print(arrayKer.shape)

res=sp.convolve2d(arrayImg, arrayKer, mode='same', boundary='fill', fillvalue=0)
print(res.shape)

plt.imshow(res)
plt.show()



#convolution is mostly integration
#y[i,j]=[sum(m=- to +inf) sum(n=- to +inf){h[m,n].x[i-m, j-n]}]

# don't try to convert the image into a matrix, its already a matrix type thingy, so just try to use this matrix
# the range should be x and y coords of the shape tuple