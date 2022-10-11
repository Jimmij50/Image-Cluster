import pickle
import cv2

from scipy.ndimage.measurements import label
from createFilterBank import create_filterbank
from getVisualWords import get_visual_words
from getImageFeatures import get_image_features
import numpy as np

filterbank=create_filterbank()


meta = pickle.load(open('../data/traintest.pkl', 'rb'))
train_imagenames = meta['train_imagenames']
train_labels= meta['train_labels']
dict = pickle.load(open('./dict.pkl', 'rb'))
dict_rand = pickle.load(open('./dict_rand.pkl', 'rb'))
head=r'C:\Study\UA\811\Assignments\assignment_mm811t\data'
for i in range(len(train_imagenames)):
    train_imagenames[i]=head+'\\'+train_imagenames[i]
 
# train_imagenames=train_imagenames[0:300]# important the sample for first 100 images
train_imagenames=train_imagenames# important the sample for first 100 images
train_features=[]
labels=[]
for i,img_path in enumerate(train_imagenames):
    #print(img_path)
    print(i)
    img=cv2.imread(img_path)
    wordmap=get_visual_words(img,dict,filterbank)
    img_feature=get_image_features(wordmap,dict.shape[0])
    train_features.append(img_feature)
    labels.append(train_labels[i])
train_features=np.array(train_features)
labels=np.array(labels)
Recon_sys={'dictionary':dict,'filterBank':filterbank,'trainFeatures':train_features
,'trainLabels':labels}
pickle.dump( Recon_sys, open( "visionHarris.pkl", "wb" ) )

Recon_sys={'dictionary':dict_rand,'filterBank':filterbank,'trainFeatures':train_features
,'trainLabels':labels}
pickle.dump( Recon_sys, open( "visionRandom.pkl", "wb" ) )
