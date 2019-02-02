import pandas as pd
import numpy as np
import jplot
import cost
import gradientDescent
#Load data
data_df = pd.read_csv('ex1data1.txt')
print(data_df)

jplot.plot(data_df)

iterations = 1500;
alpha = 0.01;

data_df=data_df.to_numpy()
print(data_df)
rows=len(data_df)
print(rows)
#X represents features
X = data_df[:,:1]
#Y represents labels
Y = data_df[:,1:2]
m = len(Y)

A = np.ones((m,1))

X=np.concatenate((A,X),axis=1)
print(X)

#For theta0 and 1 as 0 and 0

theta = np.zeros((2,1))
print(theta)

J=cost.computeCost(X,Y,theta,m)

#For theta0 and 1 as -1 and 2

theta = np.array([[-1],[2]])
print(theta)

theta = np.zeros((2,1))

theta = gradientDescent.computeGradient(X,Y,theta,m,iterations,alpha)
print(theta)

#Predict for popultation of 35,000 and 70,000
predict1 = np.array([1,3.5])
predict1 = np.matmul(predict1,theta)
print(predict1*10000)

predict1 = np.array([1,7.0])
predict1 = np.matmul(predict1,theta)
print(predict1*10000)