import numpy as np


def get_random_points(I, alpha):

    # -----fill in your implementation here --------

    h=I.shape[0]
    w=I.shape[1]
    points=[]
    for i in range(alpha):
       y=np.random.randint(0,w-1)
       x=np.random.randint(0,h-1)
       points.append((x,y))     
    # ----------------------------------------------

    return points
