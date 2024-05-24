import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#import cv2

img=mpimg.imread("Convolving Two Images/img.JPG")
img=np.array(img).reshape(img, (300, 369))
plt.imshow(img)
plt.show()

ker=mpimg.imread("Convolving Two Images/kernel.png")
ker=np.array(ker).reshape(3,3)
plt.imshow(ker)
plt.show()

pixels=[]
for i in range(360):
    for j in range(300):
        pixels.append(img[i, j])

pixels=np.array(pixels).reshape(360, 300)

conv_mat=[]
for i in range(1,358):
    for j in range(1,298):
        temp=pixels[i:i+3, j:j+3]
        pdt=np.multiply(temp, ker)
        sumAfterPdt=np.sum(pdt)
        if(sumAfterPdt>=255):
            pdt=255
        elif(sumAfterPdt<0):
            pdt=0
        conv_mat.append(sumAfterPdt)

conv_mat=np.array(conv_mat).reshape(357, 297)
print(conv_mat)
