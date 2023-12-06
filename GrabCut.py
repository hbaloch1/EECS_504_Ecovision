import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('./augmented_photos/enhanced_visible_2_2.jpg')
assert img is not None, "file could not be read, check with os.path.exists()"
mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
rect = (50,50,325,325)
cv.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]

cv.imwrite("GrabCutVisible2_2.png", img)
# img.save("image.jpg")
# plt.imshow(img)
# fig = plt.gcf()
# fig.savefig("test3_2.png")
# plt.colorbar()
# plt.show()