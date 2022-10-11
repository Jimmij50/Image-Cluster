import numpy as np
from utils import chi2dist
import pickle
import cv2
from getImageFeatures import get_image_features
from getVisualWords import get_visual_words
from getImageFeatures import get_image_features
import matplotlib.pyplot as plt 



def get_image_distance(hist1,histset,method='Euclidean'):
    distset=np.zeros(histset.shape[0])
    if method=='Euclidean':
        for i in range(histset.shape[0]):
            distset[i]=np.linalg.norm(hist1-histset[i])
    if method=='chi2':
        for i in range(histset.shape[0]):
            distset[i]=chi2dist(hist1,histset[i])
   
    return distset

def KNN(input,neighbours,labels,method='Euclidean',k=40):
    label_out=[]
    #print('neighbou',neighbours)
    label_his=np.zeros(8+1)
    #print('zero',label_his)
    distset=get_image_distance(input,neighbours,method)
    #print('distset',distset)
    ind=np.argsort(distset.flatten())
    #print(labels[ind])
    for i in range(min(k,len(ind))):
        label_his[int(labels[int(ind[i])])]+=1
    #print('label_his',label_his)
    #print(label_his.shape[0])
    ans_label=np.argsort(-label_his)[0]
    return ans_label#start from 1
    



        
imgpath= r'C:\Study\UA\811\Assignments\assignment_mm811t\python\MyCode\air1.jpg'
img=cv2.imread(imgpath)

# trained = pickle.load(open('save.pkl', 'rb'))
# meta = pickle.load(open('../data/traintest.pkl', 'rb'))
# img_features = pickle.load(open('test_img_feature.pkl', 'rb'))
# test_imagenames = meta['test_imagenames']
# test_labels=meta['test_labels']
# head=r'C:\Study\UA\811\Assignments\assignment_mm811t\data'
# for i in range(len(test_imagenames)):
#     test_imagenames[i]=head+'\\'+test_imagenames[i]
trained = pickle.load(open('visionHarris.pkl', 'rb'))
meta = pickle.load(open('../data/traintest.pkl', 'rb'))
img_features = pickle.load(open('test_img_feature_harr.pkl', 'rb'))
test_imagenames = meta['test_imagenames']
test_labels=meta['test_labels']
head=r'C:\Study\UA\811\Assignments\assignment_mm811t\data'
for i in range(len(test_imagenames)):
    test_imagenames[i]=head+'\\'+test_imagenames[i]

#print(len(test_imagenames))
#rint('sas',trained['dict'].shape[0])
# def classify(img,pkl_file,k=20):

#     wordmap=get_visual_words(img,dictionary=pkl_file['dict'],filterBank=pkl_file['filterBank'])
#     img_feature=get_image_features(wordmap,pkl_file['dict'].shape[0])
#     #print('wordmap',wordmap,wordmap.shape[0])
#     #print('f',img_feature,img_feature.shape[0])
    
#     img_class=KNN(img_feature,pkl_file['trainFeatures'],pkl_file['trainLabels'],'Euclidean',k)
#     return img_class
def classify_fe(feature,pkl_file,k=20):
    # img_class=KNN(feature,pkl_file['trainFeatures'],pkl_file['trainLabels'],'Euclidean',k)
    img_class=KNN(feature,pkl_file['trainFeatures'],pkl_file['trainLabels'],'chi2',k)
    return img_class
def eval(k):
    conf_max=np.zeros((8,8))
    for i in range(len(test_imagenames)):
        lab=classify_fe(img_features[i],trained,k)
        #print('lab',lab)
        #row for real tag
        #col for classified tag
        conf_max[int(test_labels[i])-1][int(lab)-1]+=1
        #print(conf_max)
    return conf_max
def acc(k):
    his_right=np.zeros(8)
    total=np.zeros(8)
    right=0
    tot=0
   
    for i in range(len(test_imagenames)):
        pre_lab=int(classify_fe(img_features[i],trained,k))
        rel_lab=int(test_labels[i])
        if(pre_lab==rel_lab):
            right+=1
            his_right[rel_lab-1]+=1
        total[rel_lab-1]+=1
        tot+=1
    print(his_right)
    print(total)
    print(his_right/total)
    print(right/tot)
    return right/tot,his_right/total
    #return conf_max

conf_max_set={}
acc_=[]
acc_each=np.zeros((8,40))
for i in range(40):
    k=i+1
    print('%d is processing'%k)
    conf_max_set[k]=eval(k)
    print(conf_max_set[k])
    a,b=acc(k)
    acc_.append(a)
    acc_each[:,i]=b

x=np.arange(1,41)

y=acc_each
print(y.shape)

lt=plt.plot(x,np.array(acc_),'s-',label='tot_acc')
l0=plt.plot(x,y[0],'.-',label='type1')
l1=plt.plot(x,y[1],'.-',label='type2')
l2=plt.plot(x,y[2],'.-',label='type3')
l3=plt.plot(x,y[3],'.-',label='type4')
l4=plt.plot(x,y[4],'.-',label='type5')
l5=plt.plot(x,y[5],'.-',label='type6')
l6=plt.plot(x,y[6],'.-',label='type7')
l7=plt.plot(x,y[7],'.-',label='type8')

# plt.plot(x1,y1,'ro-',x2,y2,'g+-',x3,y3,'b^-')
plt.title('Accuracy')
plt.xlabel('K')
plt.ylabel('Acc')
plt.legend()
plt.show()





# l3=plt.plot(x,np.array(acc_),   'b',label='acc')
# plt.show()

    
#pickle.dump( conf_max_set, open( "conf_max_set.pkl", "wb" ) )

