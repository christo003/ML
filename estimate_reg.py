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


rows,columns=10,10
num_lasso_path =3000

num_folder =rows*columns 
num_val=int(num_data/num_folder)

L_all,A_all,out_all,MSE_cross,idx_all,feature_all = [],[],[],[],[],[]
print('num of point in the lasso path: ',num_lasso_path)
for j in range(num_cv):
	print('num_cv : ',j+1, '/ ' , num_cv)
	I = I_all[j]
	out,mse_cross,L,A,idx_list,m,std,feature=[],[],[],[],[],0,1,[]
	for k in range(num_folder):
		print('\t num_folder : ',k+1, ' / ', num_folder)
		X_val = X[I[k*num_val:(k+1)*num_val]]
		y_val = y[I[k*num_val:(k+1)*num_val]]
		m_val = np.mean(X_val,0).reshape((1,num_feature))
		std_val = np.std(X_val,0).reshape((1,num_feature))

		#cross validation statement
		if k==0:
			X_train,y_train = X[I[(k+1)*num_val:]],y[I[(k+1)*num_val:]]
		elif k == num_folder:
			X_train,y_train=X[I[0:(k-1)*num_val]],y[I[0:(k-1)*num_val]]
		else: 
			XA,yA,XB,yB = X[I[0:(k-1)*num_val]],y[I[0:(k-1)*num_val]],X[I[(k+1)*num_val:]],y[I[(k+1)*num_val:]]
			X_train,y_train = np.concatenate((XA,XB)),np.concatenate((yA,yB))

		nval = X_train.shape[0]
		m_train,std_train  = np.mean(X_train,0).reshape((1,num_feature)),np.std(X_train,0).reshape((1,num_feature))
		tA,tL,_,_,_ = warm_start((X_train-m_train)/std_train,y_train,num_lasso_path,True,m,std)
		idx,mse = 0,sys.maxsize
		for i in range(num_lasso_path):
			current = np.mean((y_val-np.dot((X_val-m_train)/std_train,tA[i]))**2)
			if (current < mse) :
				idx,mse = i,current
		mse_cross.append(mse)
		idx_list.append(idx),out.append(tL[idx]),L.append(tL),A.append(tA),feature.append(tA[idx])
	#print(out)
	out_all.append(out),idx_all.append(idx_list),L_all.append(L),A_all.append(A),MSE_cross.append(mse_cross),feature_all.append(feature)

##########


## pour trouver le meilleure cross validation, on regarde celui qui selectionne au mieux les features
out_all,idx_all,L_all,A_all,MSE_cross,feature_all, = np.array(out_all),np.array(idx_all),np.array(L_all),np.array(A_all),np.array(MSE_cross),np.array(feature_all)

feature_select= np.median(feature_all,(0,1))
print('median feature:\n',feature_select)
dist_feature = np.mean((feature_all-feature_select)**2,2)
arr = np.mean(dist_feature,1)
best_cv = np.where(arr == arr.max())[0]
if len(best_cv)==1:
	arr= np.mean(feature_all[best_cv[0]]==feature_select,1)
	best_cv_plot=best_cv[0]
else:
	
	arr= np.mean(np.mean(feature_no_zero[best_cv]==feature_select,2),1)
	best_cv_plot = best_cv[np.random.randint(0,len(best_cv),1)]
best_fold = np.where(arr==arr.max())[0]
list_reg= out_all[best_cv,best_fold]
print('potential regularizateur: ',list_reg)
print('best cross validation : ', best_cv)
reg = np.median(list_reg)
best_cv = best_cv_plot
print('best cross validation (for plot): ',best_cv)
print('best folder in this cross val (all cv) : ', best_fold)
idx,A,L,out = idx_all[best_cv],np.array(A_all[best_cv]),np.array(L_all[best_cv]),out_all[best_cv]
med,mea = np.median(out),np.mean((out))
k=0
fig, ax_array = plt.subplots(rows, columns,squeeze=False)

for i,ax_row in enumerate(ax_array):
	for j,axes in enumerate(ax_row):
		for r in range(num_feature-1):
			axes.plot(L[k],A[k,0:-1,r+1])
		axes.plot(out[k],0,'ro',label='best on this folder')
		axes.plot(med,0,'bo',label='median')
		axes.plot(mea,0,'go',label='mean')
		axes.plot(reg,0,'ko',label='reg')
		axes.set_xlim(0,3000)
		axes.set_ylim(-1,1)
		k=k+1

fig.suptitle('CV : estimation regularizateur',fontsize=12)
plt.legend()
plt.savefig('cv_reg.png',format='png',dpi=300,bbox_inches='tight')
plt.show()

plt.figure()
plt.plot(best_cv,np.mean(MSE_cross[best_cv]),'go',label='mse')
plt.plot(np.mean(MSE_cross,1),'b',label = 'mse_cv')
plt.plot(np.mean(MSE_cross,1)+(np.std(MSE_cross,1)),'b-.',label ='mse_std_cv')
plt.plot(np.mean(MSE_cross,1)-(np.std(MSE_cross,1)),'b-.')
plt.title('mean&std of MSE throught CV')
plt.legend()
plt.savefig('mse_cv.png',format='png',dpi=300,bbox_inches='tight')
plt.show()
 
plt.figure()
plt.plot(best_cv,mea,'bo',label='med')
plt.plot(best_cv,med,'go',label='mea')
plt.plot(best_cv,reg,'ko',label='reg')

plt.plot(np.mean(out_all,1),'r',label='reg_cv')
plt.plot(np.mean(out_all,1)-(np.std(out_all,1)),'r-.',label ='reg_std_cv')
plt.plot(np.mean(out_all,1)+(np.std(out_all,1)),'r-.')
plt.title('mean&std of reg throught CV')
plt.legend()
plt.savefig('reg_cv.png',format='png',dpi=300,bbox_inches='tight')
plt.show()

alpha = A[np.argmin(np.abs(out-reg)),idx[np.argmin(np.abs(out-reg))],:]
m_train,std_train=np.mean(X,0).reshape((1,num_feature)),np.std(X,0).reshape((1,num_feature))
X_normalise=(X-m_train)/std_train
alpha,_,_,_ = lasso(X_normalise,y,alpha,reg)
residual = y- np.dot(X_normalise,alpha)
idx_no_zero = np.arange(num_feature)[alpha!=0]
np.savez('./regression_data2.npz',X=X.T[idx_no_zero].T,y=residual)
np.savez('./model_lasso.npz',alpha=alpha, reg=reg, X = X_normalise,y= y , mean= m_train, std=std_train)
print('features which lasso doesnt put to 0',idx_no_zero)
print('reg ' ,reg )

