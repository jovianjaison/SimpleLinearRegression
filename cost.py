import numpy as np

def computeCost(X,Y,theta,m):
	temp=sum((np.matmul(X,theta)-Y)*(np.matmul(X,theta)-Y)/(2*m))
	J=temp
	print(J)
	return J