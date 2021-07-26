import numpy as np
import scipy.linalg as spl
from numpy.core.fromnumeric import shape

x_data=np.ones(shape=(20,5))
y_data=np.zeros(shape=(20))
M1=spl.inv(np.matmul(x_data.transpose(),x_data))
M2=np.matmul(M1,x_data.transpose())



print(np.matmul(M2,y_data))