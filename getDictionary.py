import numpy as np
import cv2 as cv
from createFilterBank import create_filterbank
from extractFilterResponses import extract_filter_responses
from getRandomPoints import get_random_points
from getHarrisPoints import get_harris_points
from sklearn.cluster import KMeans


def get_dictionary(imgPaths, alpha, K, method):

    filterBank = create_filterbank()

    pixelResponses = np.zeros((alpha * len(imgPaths), 3 * len(filterBank)))

    for i, path in enumerate(imgPaths):
        print('-- processing %d/%d' % (i+1, len(imgPaths)))
        image = cv.imread('../data/%s' % path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)    # convert the image from bgr to rgb, OpenCV use BGR by default
        
        # -----fill in your implementation here --------
        
        if(method=='harris'):
            fun=get_harris_points
            filter_resp=extract_filter_responses(image,filterBank)
            points=fun(image,alpha,0.06)
        if(method=='random'):
            fun=get_random_points
            filter_resp=extract_filter_responses(image,filterBank)
            points=fun(image,alpha)
        for ind ,point in enumerate(points):
            row=[]
            for rep in filter_resp:
                row.append(rep[point[0],point[1]])
            rown=np.array(row)
            pixelResponses[i*alpha+ind]=rown

        # ----------------------------------------------

    dictionary = KMeans(n_clusters=K, random_state=0).fit(pixelResponses).cluster_centers_
    return dictionary
