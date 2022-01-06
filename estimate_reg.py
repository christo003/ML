import numpy as np
import sys
import matplotlib.pyplot as plt
from math import gcd
from lasso import lasso
from lasso import warm_start

data=np.load('regression_data1.npz')
y,X= data['y'],data['X']
num_data,num_feature=X.shape

I=np.arange(num_data)
np.random.shuffle(I)

nx,ny=4,4
num_lasso_path = 10000

num_folder = nx*ny
num_val=int(num_data/num_folder)
fig,axs = plt.subplots(nx,ny)
nxM=nx
out,nx=[],0
for k in range(num_folder):
	print(k)
	X_val = X[I[k*num_val:(k+1)*num_val]]
	y_val = y[I[k*num_val:(k+1)*num_val]]
	x_val_m = np.mean(X_val,1)
	x_val_std = np.std(X_val,1)

	#cross validation statement
	if k==0:
		X_train,y_train = X[I[(k+1)*num_val+1:]],y[I[(k+1)*num_val+1:]]
	elif k == num_folder:
		X_train,y_train=X[I[0:k*num_val-1]],y[I[0:k*num_val-1]]
	else: 
		XA,yA = X[I[0:(k-1)*num_val]],y[I[0:(k-1)*num_val]]
		XB,yB = X[I[(k+1)*num_val+1:]],y[I[(k+1)*num_val+1:]]
		X_train = np.concatenate((XA,XB))
		y_train = np.concatenate((yA,yB))

	nval = X_train.shape[0]
	A,L,TOL,NUM_ITER,TIME = warm_start((X_train-np.mean(X_train,1).reshape((nval,1)))/np.std(X_train,1).reshape((nval,1)),y_train,num_lasso_path)
	idx,mse = 0,sys.maxsize
	for i in range(num_lasso_path):
		current = np.mean((y_val-(x_val_std*np.dot(X_val,A[i])+x_val_m))**2)
		if (current < mse) :
			idx,mse = i,current
	if (np.mod(k,ny)==0)&(k!=0):
		nx+=1

	for i in range(num_feature-1):
		axs[nx,np.mod(k,ny)].plot(L,A.T[i+1][:-1])
	axs[nx,np.mod(k,ny)].plot(L[idx],0,'ro',label='best on this folder')
	axs[nx,np.mod(k,ny)].set_xlim(0,4000)
	axs[nx,np.mod(k,ny)].set_ylim(-1,1)

	out.append(L[idx])

med = np.median(np.array(out))
mea = np.mean(np.array(out))
alpha = A[np.argmin(np.abs(out-med))]
for i in range(nxM):
	for j in range(ny):
		axs[i,j].plot(med,0,'bo',label='median')
		axs[i,j].plot(mea,0,'go',label='mean')

fig.suptitle('CV : estimation regularizateur',fontsize=12)
plt.legend()
plt.savefig('cv_reg.png',format='png',dpi=300,bbox_inches='tight')
plt.show()
X_center=(X-np.mean(X,1).reshape((num_data,1)))/np.std(X,1).reshape((num_data,1))
reg=med
alpha,_,_,_ = lasso(X_center,y,alpha,reg)
residual = y- (np.dot(X_center,alpha)*np.std(X,1)+np.mean(X,1))
idx_no_zero = np.arange(num_feature)[alpha!=0]
np.savez('./regression_data2.npz',X=X.T[idx_no_zero].T,y=residual,alpha=alpha,reg=reg)
print('features which lasso doesnt put to 0',idx_no_zero)
