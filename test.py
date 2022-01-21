import numpy as np 
from sklearn.ensemble import RandomForestRegressor
import sys
from lasso import lasso
import csv 
data = []
with open('./yXtest.csv',newline='') as csvfile:
	spamreader = csv.reader(csvfile,delimiter = ',')
	for row in spamreader:
		data.append(row)
data=np.asmatrix(data)
data = np.array(data[1:],dtype=float)
data = data[:,1:]
y = data[:,0]
X = data[:,1:]
np.savez('./test.npz',y=y , X=X)

