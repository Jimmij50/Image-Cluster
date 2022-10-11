import numpy as np
import cv2 as cv
from numpy.core.fromnumeric import shape
from scipy import ndimage
from scipy.ndimage.filters import sobel
# from utils import imfilter

def fspecial_gaussian(size, sigma=0.5):
    m = (size-1) / 2
    y, x = np.ogrid[-m:m+1, -m:m+1]
    h = np.exp(-(x*x + y*y) / (2*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def get_harris_points(I, alpha, k=0.06):
    h=I.shape[0]
    w=I.shape[1]

    if len(I.shape) == 3 and I.shape[2] == 3:
        I = cv.cvtColor(I, cv.COLOR_RGB2GRAY)
    if I.max() > 1.0:
        I = I / 255.0
    
    # -----fill in your implementation here --------
    grad_x,grad_y=gradient(I)
    #k=0.06
    Gauss_kernal=fspecial_gaussian(3,0.5)
    #cv.filter2D()
    mat_A=cv.filter2D(grad_x*grad_x,-1,Gauss_kernal)
    mat_B=cv.filter2D(grad_x*grad_y,-1,Gauss_kernal)
    mat_C=mat_B
    mat_D=cv.filter2D(grad_y*grad_y,-1,Gauss_kernal)
    #harris_index=((grad_x*grad_x)*(grad_y*grad_y)-(grad_x*grad_y)*(grad_x*grad_y))
    #-k*((grad_x*grad_x)+(grad_y*grad_y))

    harris_index=(mat_A*mat_D)-(mat_B*mat_C)-(mat_A+mat_B)*(mat_A+mat_B)

    k_th_max_harris=max(0,np.partition(harris_index.flatten(), -1*alpha)[-1*alpha])
    flag_index=np.where(harris_index>=k_th_max_harris,1,0)
    points=[]
    for i in range(h):
        for j in range(w):
            if (flag_index[i][j]==1):
                points.append((i,j))


    
    # ----------------------------------------------
    
    return points

def gradient(I):# I in grey scale
    
    #using sobel to calculate grdient
    #I = cv.cvtColor(I, cv.COLOR_RGB2GRAY)
    #I = cv.copyMakeBorder(I, 1, 1, 1, 1, cv.BORDER_REPLICATE ) #padding for 3x3
    kernal_y=np.zeros((3,3))
    kernal_y[0,0]=-1
    kernal_y[0,1]=-2
    kernal_y[0,2]=-1

    kernal_y[2,0]=1
    kernal_y[2,1]=2
    kernal_y[2,2]=1

    kernal_x=np.zeros((3,3))
    kernal_x[0,0]=-1
    kernal_x[1,0]=-2
    kernal_x[2,0]=-1

    kernal_x[0,2]=1
    kernal_x[1,2]=2
    kernal_x[2,2]=1
    print(I.shape)
    print(type(I))

    grad_map_x=cv.filter2D(I,-1,kernal_x)
    grad_map_y=cv.filter2D(I,-1,kernal_y)
    
    print(grad_map_x.shape)
    return grad_map_x,grad_map_y
if __name__=='__main__':
    path= r'C:\Study\UA\811\Assignments\assignment_mm811t\python\r1.jpg'
    img1=cv.imread(path)
    print(img1.dtype)
    a,b=gradient(img1)
    points=get_harris_points(img1,500,0.06)
    for i in points:
        cv.circle(img1,(i[1],i[0]),1,(255,0,0),1)
    while(1):
        cv.imshow('ans',img1)
        cv.imshow('b',b)
        if cv.waitKey(10) == ord('Q'):
            break

# def top_k(arr): #return the k-th max value's index
#     ind=0
#     return ind
