import numpy as np
import sys
import matplotlib.pyplot as plt
from math import gcd
from lasso import lasso
from lasso import warm_start

data=np.load('regression_data1.npz')
y,X= data['y'],data['X']
num_data,num_feature=X.shape


logscale =lambda k,m: np.exp(np.log(1/(10*m))+k*(np.log(m)-np.log(1/(10*m)))/(m-1))


rows,columns=2,2
num_lasso_path =np.int(np.max(np.dot(X.T,y)))

		
tL = [logscale(i,num_lasso_path)  for i in range(num_lasso_path,0,-1)]
idx,gsure = 0,sys.maxsize
a=0
print('num_lasso',int(num_lasso_path))
m,std = np.mean(X,0).reshape((1,num_feature)),np.std(X,0).reshape((1,num_feature))
Xc = (X-m)/std
alpha= np.zeros(num_feature)
for i in range(int(num_lasso_path)):
	alpha,_,_,_ = lasso(Xc,y,np.zeros(num_feature),tL[i])
	mu = np.dot(Xc,alpha)
	current = num_data*((mu-y)**2).sum()/((1-np.linalg.matrix_rank(X[:,np.arange(num_feature)[0!=alpha]]))**2)
	print('gsure : ',current, 'reg : ', tL[i], ' on ' ,i, ' / ', num_lasso_path)

	if (current < gsure) :
		idx,gsure = i,current
		a=alpha
		reg = tL[i]

print('reg' , reg)
print('smaleest gsure : ' ,gsure)
residual = y-np.dot(Xc,a)	
print('idx_no_zero =' , np.arange(num_feature)[0!=a])
np.savez('data_regression3.npz',X=X,y=residual)
