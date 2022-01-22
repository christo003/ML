import numpy as np
import sys
from lasso import lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
data = np.load('regression_data1.npz')
X,y = data['X'],data['y']
num_data,num_feature = X.shape 
parameters=np.load('parameters.npz',allow_pickle=True)
la,param_forest,m_lasso,std_lasso,reg_lasso=parameters['model_lasso'],parameters['param_forest'],parameters['m_lasso'],parameters['std_lasso'],parameters['reg_lasso']
reg_ridge = parameters['reg_ridge']
alpha = np.zeros(num_feature)
param_forest = param_forest.item()
#param_forest['criterion']='poisson'
num =10
num_cv = 10
num_fold=int(num_data/num_cv)
I=np.arange(num_data)

out_val_ridge,out_val_RERFs,out_val_lasso,out_train_ridge,out_train_RERFs,out_train_lasso = [],[],[],[],[],[]
for i in range(num):
	print('num_cv : ',i+1, '/ ' , num_cv) 
	np.random.shuffle(I)
	val_ridge,val_RERFs,val_lasso,train_ridge,train_RERFs,train_lasso = [],[],[],[],[],[]
	for j in range(num_cv):
	
		X_val = X[I[j*num_fold:(j+1)*num_fold]]
		y_val = y[I[j*num_fold:(j+1)*num_fold]]
		#cross validation statement
		if j==0:
			X_train,y_train = X[I[(j+1)*num_fold:]],y[I[(j+1)*num_fold:]]
		elif j == num_fold:
			X_train,y_train=X[I[0:(j-1)*num_fold]],y[I[0:(j-1)*num_fold]]
		else:
			XA,yA,XB,yB = X[I[0:(j-1)*num_fold]],y[I[0:(j-1)*num_fold]],X[I[(j+1)*num_fold:]],y[I[(j+1)*num_fold:]]
			X_train,y_train = np.concatenate((XA,XB)),np.concatenate((yA,yB))
		nval = X_train.shape[0]

		#m_train,std_train  = np.mean(X_train,0).reshape((1,num_feature)),np.std(X_train,0).reshape((1,num_feature))

		num_train,num_val = X_train.shape[0],X_val.shape[0]
		#X_train = (X_train - m_train ) / std_train
		
		#X_val = (X_val - m_train) / std_train
		v_val = ((y_val-y_val.mean())**2).sum()
		v_train = ((y_train-y_train.mean())**2).sum()
		
		#a,_,_,_ = lasso(X_train,y_train,alpha,reg_lasso)
		la = Lasso(alpha = reg_lasso)
		la.fit(X_train,y_train)
		val_pred_lasso = la.predict(X_val)#np.dot(X_val,a)
		train_pred_lasso=la.predict(X_train)#np.dot(X_train,a)	

		val_lasso.append(1-(((y_val-val_pred_lasso)**2).sum()/v_val))
		train_lasso.append(1-(((y_train-train_pred_lasso)**2).sum()/v_train))

		
		ridge= Ridge(alpha=reg_ridge)
		ridge.fit(X_train,y_train)
		val_ridge.append(ridge.score(X_val,y_val))
		train_ridge.append(ridge.score(X_train,y_train))
		
		train_forest = X_train
		val_forest = X_val
		target_forest =y_train-train_pred_lasso
		
		rfr = RandomForestRegressor(**param_forest)
		rfr.fit(train_forest,target_forest)
		val_pred_RERFs = val_pred_lasso + rfr.predict(val_forest)
		train_pred_RERFs = train_pred_lasso + rfr.predict(train_forest)
		val_RERFs.append(1-(((y_val-val_pred_RERFs)**2).sum()/v_val))
		train_RERFs.append(1-(((y_train-train_pred_RERFs)**2).sum()/v_train))
	out_val_ridge.append(np.median(val_ridge)),out_val_RERFs.append(np.median(val_RERFs)),out_val_lasso.append(np.median(val_lasso)),out_train_ridge.append(np.median(train_ridge)),out_train_RERFs.append(np.median(train_RERFs)),out_train_lasso.append(np.median(train_lasso)) 

	
val_ridge,val_RERFs,val_lasso,train_ridge,train_RERFs,train_lasso =out_val_ridge,out_val_RERFs,out_val_lasso,out_train_ridge,out_train_RERFs,out_train_lasso
plt.figure()
mev_r,mev_R,mev_l=np.mean(val_ridge),np.mean(val_RERFs),np.mean(val_lasso)
mav_r,mav_R,mav_l=np.median(val_ridge),np.median(val_RERFs),np.median(val_lasso)
plt.plot(val_ridge,'b-',label='val_ridge')
plt.plot(val_RERFs,'r-',label='val RERFs')
#plt.plot(val_lasso,'g-',label='val lasso')
plt.plot(train_RERFs,'r:',label='train_RERfs')
#plt.plot(train_lasso,'g.',label='train_lasso')
plt.plot(train_ridge,'b:',label='train_ridge')
#plt.plot([0,num_cv],[mav_l,mav_l],'g-.',label='median val lasso')
plt.plot([0,num_cv],[mav_R,mav_R],'m--',label='median val RERFs')
plt.plot([0,num_cv],[mav_r,mav_r],'c--',label='median val ridge')
plt.legend()
plt.savefig('compare.png',format='png',dpi=300,bbox_inches='tight')
plt.show()
print('mean ridg',mev_r)
print('mean RERF',mev_R)
print('median ridg',mav_r)
print('median RERFs',mav_R)
print('reg lasso' ,reg_lasso)
