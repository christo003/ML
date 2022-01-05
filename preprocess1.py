import numpy as np
import csv

out=[]
with open('./regression_data.csv',newline='') as csvfile:
	spamreader= csv.reader(csvfile,delimiter = ',')
	for row in spamreader: 
		out.append(row)
out = np.asmatrix(out)
out =np.array(out[1:],dtype=float)
y = out[:,1]
X = out[:,2:]
np.savez('./regression_data.npz',y=y,X=X)
