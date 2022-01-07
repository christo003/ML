import numpy as np
import sys
import matplotlib.pyplot as plt
from math import gcd
from lasso import lasso
from lasso import warm_start

data=np.load('regression_data1.npz')
y,X= data['y'],data['X']
num_data,num_feature=X.shape

num_cv  = 10
I_all=np.array([np.arange(num_data) for i in range(num_cv)])
for i in range(num_cv):
	np.random.shuffle(I_all[i])


rows,columns=3,3
num_lasso_path = 100

num_folder =rows*columns 
num_val=int(num_data/num_folder)
L_all = []
A_all = []
out_all = []
MSE_cross = []
idx_all = []
for j in range(num_cv):
	print('num_cv : ',j+1, '/ ' , num_cv)
	I = I_all[j]
	out=[]
	mse_cross = 0
	L=[]
	A=[]
	idx_list=[]
	for k in range(num_folder):
		print('\t num_folder : ',k+1, ' / ', num_folder)
		X_val = X[I[k*num_val:(k+1)*num_val]]
		y_val = y[I[k*num_val:(k+1)*num_val]]
		m_val = np.mean(X_val,1)
		std_val = np.std(X_val,1)

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
		m_train = np.mean(X_train,1).reshape((nval,1))
		std_train = np.std(X_train,1).reshape((nval,1))
		tA,tL,_,_,_ = warm_start((X_train-m_train)/std_train,y_train,num_lasso_path)
		idx,mse = 0,sys.maxsize
		for i in range(num_lasso_path):
			current = np.mean((y_val-(std_train*np.dot(X_val,tA[i])+m_train))**2)
			if (current < mse) :
				idx,mse = i,current
		mse_cross+=mse
		idx_list.append(idx)
		out.append(tL[idx])
		L.append(tL)
		A.append(tA)
	out_all.append(out)
	idx_all.append(idx_list)
	L_all.append(L)
	A_all.append(A)	
	MSE_cross.append(mse_cross)

##########

best_cv = np.argmin(MSE_cross)
idx = idx_all[best_cv]
A=np.array(A_all[best_cv])
L=np.array(L_all[best_cv])
out = np.array(out_all[best_cv])
med = np.median(np.array(out))
mea = np.mean(np.array(out))


k=0
fig, ax_array = plt.subplots(rows, columns,squeeze=False)

for i,ax_row in enumerate(ax_array):
	for j,axes in enumerate(ax_row):
		for r in range(num_feature-1):
			axes.plot(L[k],A[k].T[r+1][:-1])
		axes.plot(out[k],0,'ro',label='best on this folder')
		axes.plot(med,0,'bo',label='median')
		axes.plot(mea,0,'go',label='mean')
		axes.set_xlim(0,4000)
		axes.set_ylim(-1,1)
		k=k+1
fig.suptitle('CV : estimation regularizateur',fontsize=12)
plt.legend()
plt.savefig('cv_reg.png',format='png',dpi=300,bbox_inches='tight')
plt.show()

plt.figure()
plt.plot(MSE_cross)
plt.savefig('mse_cv_lasso.png',format='png',dpi=300,bbox_inches='tight')
plt.show()

alpha = A[np.argmin(np.abs(out-med)),idx[np.argmin(np.abs(out-med))],:]
m,std=np.mean(X,1).reshape((num_data,1)),np.std(X,1).reshape((num_data,1))
X_center=(X-m)/std
reg=med
alpha,_,_,_ = lasso(X_center,y,alpha,reg)
residual = y- (np.dot(X_center,alpha)*std+m)
idx_no_zero = np.arange(num_feature)[alpha!=0]
np.savez('./regression_data2.npz',X=X.T[idx_no_zero].T,y=residual)
np.savez('./model_lasso.npz',X=X_center,y=y,m=m,std=std,alpha=alpha,reg=reg)
print('features which lasso doesnt put to 0',idx_no_zero)
print('reg ' ,reg )

