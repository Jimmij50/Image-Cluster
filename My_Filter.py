import cv2
import numpy as np
import skimage
import RGB2Lab
path= r'C:\Study\UA\811\Assignments\assignment_mm811t\python\MyCode\air1.jpg'

img1=cv2.imread(path)
img1_lab=RGB2Lab.rgb2lab(img1)
h=img1.shape[0]
w=img1.shape[1]

# vis = np.zeros((384, 836), np.float32)
# h,w = vis.shape
# vis2 = cv.CreateMat(h, w, cv.CV_32FC3)
# vis0 = cv.fromarray(vis)
# cv.CvtColor(vis0, vis2, cv.CV_GRAY2BGR)


# vis1=cv2.CreateMath(h,w,cv2.cv_32F1)
# vis2=cv2.CreateMath(h,w,cv2.cv_32F1)
# vis3=cv2.CreateMath(h,w,cv2.cv_32F1)
vis1=img1_lab[...,0]
vis2=img1_lab[...,1]
vis3=img1_lab[...,2]
print(img1_lab)
while(1):
    cv2.imshow('air1',img1)
    cv2.imshow('air1_lab',img1_lab)
    cv2.imshow('vis1',vis1)
    cv2.imshow('vis2',vis2)
    cv2.imshow('vis3',vis3)
    if cv2.waitKey(10) == ord('Q'):
        break

