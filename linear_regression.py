from typing import Text
import numpy as np


def h(x): # hypothesis function
    theta_0=0
    theta_1=1
    return theta_0+x*theta_1

def cost(data):
    x=data[:,0].flatten()
    y=data[:,1].flatten()
    m=len(x) # size of training set
    return (np.sum((h(x)-y)**2))*(1/(2*m))

