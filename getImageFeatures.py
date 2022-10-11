import numpy as np
import pickle
from getVisualWords import get_visual_words
def get_image_features(wordMap, dictionarySize):

    # -----fill in your implementation here --------


    

    h=np.zeros(dictionarySize)
    height=wordMap.shape[0]
    width=wordMap.shape[1]
    for i in range(height):
        for j in range(width):
            h[int(wordMap[i][j])]=h[int(wordMap[i][j])]+1
    for i in range(h.size):
        h[i]=int(h[i])
    h=100*h/(height*width)
    #print('h',h)
    # ----------------------------------------------
    return h

