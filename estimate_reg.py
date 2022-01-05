import numpy as np
import sys
import matplotlib.pyplot as plt
from math import gcd
from lasso import lasso
from lasso import warm_start

data=np.load('regression_data.npz')
y,X= data['y'],data['X']
num_data,num_feature=X.shape

I=np.arange(num_data)
np.random.shuffle(I)

nx,ny=4,4
num_lasso_path = 1000

num_folder = nx*ny
num_val=int(num_data/num_folder)
fig,axs = plt.subplots(nx,ny)
nxM=nx
out,nx=[],0
for k in range(num_folder):
	X_val = X[I[k*num_val:(k+1)*num_val]]
	y_val = y[I[k*num_val:(k+1)*num_val]]
	x_val_m = np.mean(X_val,1)
	x_val_std = np.std(X_val,1)
	if k==0:
		X_train,y_train = X[I[(k+1)*num_val+1:]],y[I[(k+1)*num_val+1:]]
	elif k == num_folder:
		X_train,y_train=X[I[0:k*num_val-1]],y[I[0:k*num_val-1]]
	else: 
		XA,yA = X[I[0:(k-1)*num_val]],y[I[0:(k-1)*num_val]]
		XB,yB = X[I[(k+1)*num_val+1:]],y[I[(k+1)*num_val+1:]]
		X_train = np.concatenate((XA,XB))
		y_train = np.concatenate((yA,yB))

	
	A,L,TOL,NUM_ITER,TIME = warm_start(X_train,y_train,num_lasso_path)
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
	
	#plt.legend()
	out.append(L[idx])

med = np.median(np.array(out))
mea = np.mean(np.array(out))
for i in range(nxM):
	for j in range(ny):
		axs[i,j].plot(med,0,'bo',label='median')
		axs[i,j].plot(mea,0,'go',label='mean')
plt.legend()
plt.savefig('lasso_path.png',format='png',dpi=300,bbox_inches='tight')
plt.show()
reg=med
np.savez('./lambda.npz',reg=reg)
