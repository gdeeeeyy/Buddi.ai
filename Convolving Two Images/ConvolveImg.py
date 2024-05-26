import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img=mpimg.imread("Convolving Two Images/img.JPG")
sp1=img.shape
print(img)
plt.imshow(img)
#plt.show()

ker=mpimg.imread("Convolving Two Images/kernel.png")
plt.imshow(ker)
#plt.show()

#np.convolve

#convolution is mostly integration
#y[i,j]=[sum(m=- to +inf) sum(n=- to +inf){h[m,n].x[i-m, j-n]}]

# don't try to convert the image into a matrix, its already a matrix type thingy, so just try to use this matrix
# the range should be x and y coords of the shape tuple