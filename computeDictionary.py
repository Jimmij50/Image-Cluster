import pickle
from getDictionary import get_dictionary
import cv2

meta = pickle.load(open('../data/traintest.pkl', 'rb'))
train_imagenames = meta['train_imagenames']




# -----fill in your implementation here --------
head=r'C:\Study\UA\811\Assignments\assignment_mm811t\data'

a=cv2.imread(head+'\\'+train_imagenames[0])
# train_imagenames=train_imagenames[0:10]
for i in train_imagenames:
    i=head+'\\'+i
dict =get_dictionary(train_imagenames,200,500,'harris')
print('dict is ready to save')
with open('dict.pkl', 'wb') as file:
    pickle.dump(dict, file)
dict_rand =get_dictionary(train_imagenames,200,500,'random')
print('dict_rand is ready to save')
with open('dict_rand.pkl', 'wb') as file:
    pickle.dump(dict_rand, file)


# ----------------------------------------------



