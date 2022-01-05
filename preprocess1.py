import numpy as np
import csv

out=[]
with open('./regression_data.csv',newline='') as csvfile:
	spamreader= csv.reader(csvfile,delimiter = ',')
	for row in spamreader: 
		out.append(row)
out = np.asmatrix(out)
out = out[1:]
out = out[:,1:]
out = np.array(out,dtype=float)
np.save('./regression_data',out)
