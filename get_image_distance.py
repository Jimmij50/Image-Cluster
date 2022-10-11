import numpy as np
from utils import chi2dist
def get_image_distance(hist1,hist2,method='Euclidean'):
    dist=1000000.0
    if method=='Euclidean':
        dist=np.linalg.norm(hist1-hist2)
    if method=='chi2':
        dist=chi2dist(hist1,hist2)
    return dist
        
if __name__=='__main__':
    a=np.array([1,2,1])
    b=np.array([1,2,3])
    print(get_image_distance(a,b,method='chi2'))