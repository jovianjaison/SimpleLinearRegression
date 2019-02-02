import numpy as np

def computeGradient(X,Y,theta,m,iterations,alpha):
	for i in range(1,iterations):
		x = np.matmul(X,theta)-Y
		theta0 = theta[0][0]-((alpha/m)*sum(x))
		theta1 = theta[1][0]-((alpha/m)*sum(np.matmul(X[:,1],x)))
		theta[0][0] = theta0
		theta[1][0] = theta1
	return theta
