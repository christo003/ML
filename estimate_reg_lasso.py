import numpy as np
import sys
import matplotlib.pyplot as plt
from math import gcd
from lasso import lasso
from lasso import warm_start
from sklearn.linear_model import Lasso

data=np.load('regression_data1.npz')
y,X= data['y'],data['X']
num_data,num_feature=X.shape

num_cv  =3 
I_all=np.array([np.arange(num_data) for i in range(num_cv)])
for i in range(num_cv):
	np.random.shuffle(I_all[i])

logscale =lambda k,m: np.exp(np.log(1/(10*m))+k*(np.log(m)-np.log(1/(10*m)))/(m-1))


rows,columns=2,2
num_lasso_path = 500

num_folder =rows*columns 
num_val=int(num_data/num_folder)

out_all,L_all,GSURE_cross,idx_all,feature_all = [],[],[],[],[]
cv_med = []
idx_gsure_med=[]
for j in range(num_cv):
	print('num_cv : ',j+1, '/ ' , num_cv)
	I = I_all[j]
	out,gsure_cross,L,A,idx_list,m,std,feature=[],[],[],[],[],0,1,[]
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

		num_train = X_train.shape[0]
		m_train,std_train  = np.mean(X_train,0).reshape((1,num_feature)),np.std(X_train,0).reshape((1,num_feature))
		#num_lasso_path=np.max(np.dot(X_train.T,y_train))
		lambda_max=np.max(np.dot(2*X_train.T,y_train))/num_train
		tL = [logscale(int(i),int(lambda_max))  for i in np.linspace(0,lambda_max,num_lasso_path)]
		idx,gsure = 0,sys.maxsize
		a=0
		print('num_lasso',int(num_lasso_path))
		i=0
		la = Lasso(alpha=tL[i])
		la.fit(X_train,y_train)
		tA=la.coef_

		while (i <(int(num_lasso_path)))&(np.abs(tA).sum()!=0):
			#print(tL[i],tA)
		#	tA,_,_,_ = lasso((X_train-m_train)/std_train,y_train,np.zeros(num_feature),tL[i])
		#	mu = np.dot((X_val-m_train)/std_train,tA)
			current = num_val*((la.predict(X_val)-y_val)**2).sum()/((1-np.linalg.matrix_rank(X_val[:,np.arange(num_feature)[0!=tA]]))**2)
			print('gsure : ',current, 'reg : ', tL[i], ' on ' ,i, ' / ', num_lasso_path)

			if (current < gsure) :
				idx,gsure = i,current
				a=tA
			i+=1
			la = Lasso(alpha=tL[i])
			la.fit(X_train,y_train)
			tA=la.coef_
		gsure_cross.append(gsure)
		idx_list.append(idx),out.append(tL[idx]),L.append(tL),feature.append(a)
	idx_gsure_med.append(np.argmin(np.abs(gsure_cross-np.median(gsure_cross))))
	out_all.append(out),idx_all.append(idx_list),L_all.append(L),GSURE_cross.append(gsure_cross[idx_gsure_med[-1]]),feature_all.append(feature)
	

##########
best_cv = np.argmin(GSURE_cross)
idx_reg_cv = idx_gsure_med[best_cv]
out_all=np.array(out_all)
reg =out_all[best_cv,idx_reg_cv]
out_all,idx_all,L_all,GSURE_cross,feature_all, = np.array(out_all),np.array(idx_all),np.array(L_all),np.array(GSURE_cross),np.array(feature_all)
print('reg found',reg)


idx,A,L,out = idx_all[best_cv],np.array(feature_all[best_cv]),np.array(L_all[best_cv]),out_all[best_cv]
med,mea = np.median(out),np.mean((out))
k=0
#fig, ax_array = plt.subplots(rows, columns,squeeze=False)
#
#for i,ax_row in enumerate(ax_array):
#	for j,axes in enumerate(ax_row):
#		#axes.plot(L_all[best_cv,k],A[k])
#		axes.plot(out[k],0,'ro',label='best on this folder')
#		axes.plot(med,0,'bo',label='median')
#		axes.plot(mea,0,'go',label='mean')
#		axes.plot(reg,0,'ko',label='reg')
#		axes.set_xlim(0,3000)
#		axes.set_ylim(-1,1)
#		k=k+1

#ig.suptitle('CV : estimation regularizateur',fontsize=12)
#plt.legend()
#lt.savefig('cv_reg.png',format='png',dpi=300,bbox_inches='tight')
#plt.show()

#plt.figure()
#plt.plot(best_cv,np.mean(GSURE_cross[best_cv]),'go',label='gsure')
#plt.plot(np.mean(GSURE_cross,1),'b',label = 'gsure_cv')
#plt.plot(np.mean(GSURE_cross,1)+(np.std(GSURE_cross,1)),'b-.',label ='gsure_std_cv')
#plt.plot(np.mean(GSURE_cross,1)-(np.std(GSURE_cross,1)),'b-.')
#plt.title('mean&std of GSURE throught CV')
#plt.legend()
#plt.savefig('gsure_cv.png',format='png',dpi=300,bbox_inches='tight')
#plt.show()
 
#plt.figure()
#plt.plot(best_cv,mea,'bo',label='med')
#plt.plot(best_cv,med,'go',label='mea')
#plt.plot(best_cv,reg,'ko',label='reg')

#plt.plot(np.mean(out_all,1),'r',label='reg_cv')
#plt.plot(np.mean(out_all,1)-(np.std(out_all,1)),'r-.',label ='reg_std_cv')
#plt.plot(np.mean(out_all,1)+(np.std(out_all,1)),'r-.')
#plt.title('mean&std of reg throught CV')
#plt.legend()
#plt.savefig('reg_cv.png',format='png',dpi=300,bbox_inches='tight')
#plt.show()

alpha = A[idx_reg_cv]
m_train,std_train=np.mean(X,0).reshape((1,num_feature)),np.std(X,0).reshape((1,num_feature))
X_normalise=(X-m_train)/std_train
alpha,_,_,_ = lasso(X_normalise,y,np.zeros(num_feature),reg)
residual = y- np.dot(X_normalise,alpha)
idx_no_zero = np.arange(num_feature)[alpha!=0]
np.savez('./regression_data2.npz',X=X,y=residual)
np.savez('./model_lasso.npz',alpha=alpha, reg=reg, X = X_normalise,y= y , mean= m_train, std=std_train)
print('features which lasso doesnt put to 0',idx_no_zero)
#print('reg ' ,reg )
#plt.figure()
#plt.plot(REG)
#plt.savefig('reg_cv_lasso_path',format='png',dpi=300,bbox_inches='tight')
#plt.show()
