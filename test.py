import pickle
from getVisualWords import get_visual_words
from getImageFeatures import get_image_features
import cv2
import numpy as np
pkl_file_harr= pickle.load(open('visionHarris.pkl', 'rb'))
pkl_file_rand= pickle.load(open('visionRandom.pkl', 'rb'))
# pkl_file= pickle.load(open('save.pkl', 'rb'))
meta = pickle.load(open('../data/traintest.pkl', 'rb'))
# train_imagenames = meta['train_imagenames']
# print(meta.keys())
# trained = pickle.load(open('save.pkl', 'rb'))
# meta = pickle.load(open('../data/traintest.pkl', 'rb'))
test_imagenames = meta['test_imagenames']
# test_labels=meta['test_labels']
# dict=trained['dict']
# print(len(train_imagenames))
# m= pickle.load(open('conf_max_set.pkl', 'rb'))
# print(m[2])

#Generate Feature pickle

test_img_feature_harr=[]
test_img_feature_rand=[]
head=r'C:\Study\UA\811\Assignments\assignment_mm811t\data'
for i in range(len(test_imagenames)):
    test_imagenames[i]=head+'\\'+test_imagenames[i]
# img=cv2.imread(test_imagenames[0])
# while(1):
#     cv2.imshow('img',img)
#     cv2.waitKey(10)
for i in range(len(test_imagenames)):
    img=cv2.imread(test_imagenames[i])
    wordmap=get_visual_words(img,dictionary=pkl_file_harr['dictionary'],filterBank=pkl_file_harr['filterBank'])
    img_feature=get_image_features(wordmap,pkl_file_harr['dictionary'].shape[0])
    test_img_feature_harr.append(img_feature)
    print(i)
for i in range(len(test_imagenames)):
    img=cv2.imread(test_imagenames[i])
    wordmap=get_visual_words(img,dictionary=pkl_file_rand['dictionary'],filterBank=pkl_file_rand['filterBank'])
    img_feature=get_image_features(wordmap,pkl_file_rand['dictionary'].shape[0])
    test_img_feature_rand.append(img_feature)
    print(i)
# ttt=np.array(test_img_feature)
# print(ttt.shape)
with open('test_img_feature_harr.pkl', 'wb') as file:
    pickle.dump(test_img_feature_harr, file)
with open('test_img_feature_rand.pkl', 'wb') as file:
    pickle.dump(test_img_feature_rand, file)




# test_fea = pickle.load(open('../data/traintest.pkl', 'rb'))