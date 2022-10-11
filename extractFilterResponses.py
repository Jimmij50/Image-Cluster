import cv2 as cv
import numpy as np
from createFilterBank import create_filterbank
from RGB2Lab import rgb2lab
from utils import *


def extract_filter_responses(I, filterBank):

    I = I.astype(np.float64)
    if len(I.shape) == 2:
        I = np.tile(I, (3, 1, 1))

    # -----fill in your implementation here --------
    I=rgb2lab(I)
    filterResponses=[]
   
    for filter in filterBank:
        for i in range(3):
            filterResponses.append(cv.filter2D(I[...,i],-1,filter))


    # ----------------------------------------------
    
    return filterResponses
if (__name__=='__main__'):

    path= r'C:\Study\UA\811\Assignments\assignment_mm811t\python\1.jpg'
    img1=cv.imread(path)
    cv.imshow('ori',img1)
    ans=extract_filter_responses(img1,create_filterbank())
    print(len(ans))
    mer=np.dstack((ans[3],ans[4],ans[5]))
    mer=np.float32(mer)
    print(mer.shape)
    mer=cv.cvtColor(mer,cv.COLOR_LAB2RGB)
    while(1):
        img1_lab=rgb2lab(img1)
        #img1=rgb2lab(img1)
        vis1=img1_lab[...,0]
        vis2=img1_lab[...,1]
        vis3=img1_lab[...,2]
        nor1=cv.normalize(ans[3],cv.NORM_MINMAX)*255
        nor2=cv.normalize(ans[4],cv.NORM_MINMAX)*255
        nor3=cv.normalize(ans[5],cv.NORM_MINMAX)*255
    # print(nor1)
        cv.imshow('ans1',nor1)
        cv.imshow('ans2',nor2)
        cv.imshow('ans3',nor3)
        cv.imshow('vis1',vis1)
        cv.imshow('vis2',vis2)
        cv.imshow('vis3',vis3)
        cv.imshow('mer',mer)
        if cv.waitKey(10) == ord('Q'):
            break
