import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.signal as sp

#opening the image
img=Image.open("Convolving Two Images/img.jpg")
#converting the image to b&w
imgBw=img.convert("1")
#plotting the image
plt.imshow(imgBw)
plt.show()

#converting the image into a numpy array
arrayImg=np.array(imgBw, dtype=np.float64)
print(arrayImg.shape)

#opening the kernel image
kerImg=Image.open("Convolving Two Images/ker2.png")
#converting the kernel to b&w
kerBw=kerImg.convert("1")
#plotting the kernel
plt.imshow(kerBw)
plt.show()

#converting the kernel image to an array
arrayKer=np.array(kerBw, dtype=np.float64)
print(arrayKer.shape)

#convolving the image and the kernel
res=sp.convolve2d(arrayImg, arrayKer, mode='valid', boundary='wrap')
print(res.shape)

#plotting the output of the convolution
plt.imshow(res)
plt.show()