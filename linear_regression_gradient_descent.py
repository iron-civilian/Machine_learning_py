from typing import Text
import numpy as np
import matplotlib.pyplot as plt

def h(x,theta_0,theta_1): # hypothesis function
    return theta_0+x*theta_1

def cost(x,y,theta_0,theta_1):
    m=len(x) # size of training set
    return (np.sum((h(x,theta_0,theta_1)-y)**2))*(1/(2*m))



def gradient_descent(x,y,theta_0,theta_1):
    alpha=1e-2
    m=len(x)
    temp0=theta_0 - (alpha/m)*np.sum(h(x,theta_0,theta_1)-y) 
    temp1=theta_1 - (alpha/m)*np.sum((h(x,theta_0,theta_1)-y)*x)
    return temp0,temp1

x=np.linspace(1,10,50)
y=x*2+3

theta0=0
theta1=0
count=5000
cost_list=[]
for i in range(count):
    theta0,theta1=gradient_descent(x,y,theta0,theta1)
    cost_list.append(cost(x,y,theta0,theta1))
print("{theta0:.2f},{theta1:.2f}".format(theta0=theta0,theta1=theta1))
