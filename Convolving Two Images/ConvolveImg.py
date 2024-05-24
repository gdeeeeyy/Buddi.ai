import numpy as np
import matplotlib.pyplot as plt
import cv2

img=cv2.imread("""Image path""")
plt.imshow(img)
plt.show()

ker=cv2.imread("""Image kernel path""")
plt.imshow(ker)
plt.show()