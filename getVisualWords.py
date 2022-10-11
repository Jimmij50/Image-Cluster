import numpy as np
from scipy.spatial.distance import cdist
from extractFilterResponses import extract_filter_responses
from getDictionary import get_dictionary
import pickle
import skimage.color
import cv2
import createFilterBank

def get_visual_words(I, dictionary, filterBank):

    # -----fill in your implementation here --------

    h=I.shape[0]
    w=I.shape[1]
    wordMap=np.ones((h,w))
    filterresp=extract_filter_responses(I,filterBank)

    # size=dictionary.cluster_centers_.shape[0]
    k_size=dictionary.shape[0]
    feature_size=dictionary.shape[1]
    #print(dictionary.shape)
    #print(dictionary[0].shape)
    for i in range(h):
            for j in range(w):
                vector=[]
                for res in filterresp:
                    vector.append(res[i][j])
                vector=np.array(vector)
                    #dis=cdist(vector,dictionary[k],'euclidean')
                dis_arr=cdist(vector.reshape(-1,feature_size),dictionary,'euclidean')
                # print('dis arr',dis_arr.shape)
                wordMap[i][j]=np.argmin(dis_arr)
    #print('end')


                        
                
    # ----------------------------------------------

    return wordMap

#for test
if __name__=='__main__':
    # path=r'C:\Study\UA\811\Assignments\assignment_mm811t\python\d3.jpg'
    path=r'C:\Users\jimmi\OneDrive\Pictures\m3.jpg'
    img=cv2.imread(path)

    img=cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2)))

    dict = pickle.load(open('./dict.pkl', 'rb'))
    dict_rand = pickle.load(open('./dict_rand.pkl', 'rb'))
    wordmap=get_visual_words(img,dict,createFilterBank.create_filterbank())
    wordmap_rand=get_visual_words(img,dict_rand,createFilterBank.create_filterbank())
    while(1):
        cv2.imshow('ori',img)
        cv2.imshow('harris',skimage.color.label2rgb(wordmap))
        cv2.imshow('random',skimage.color.label2rgb(wordmap_rand))
        if cv2.waitKey(10) == ord('Q'):
            break



