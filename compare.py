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
reg_lasso,param_forest,baseline_reg_lasso=parameters['RERFs_param_lasso'],parameters['RERFs_param_forest'],parameters['reg_lasso']
reg_ridge = parameters['reg_ridge']
param_forest = param_forest.item()
num =10
num_cv = 10
num_fold=int(num_data/num_cv)
I=np.arange(num_data)

out_val_ridge,out_val_RERFs,out_val_lasso,out_train_ridge,out_train_RERFs,out_train_lasso = [],[],[],[],[],[]
out_val_lasso_baseline=[]
out_train_lasso_baseline=[]
for i in range(num):
	print('num_cv : ',i+1, '/ ' , num_cv) 
	np.random.shuffle(I)
	val_ridge,val_RERFs,val_lasso,train_ridge,train_RERFs,train_lasso = [],[],[],[],[],[]
	val_lasso_baseline=[]
	train_lasso_baseline=[]
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


		num_train,num_val = X_train.shape[0],X_val.shape[0]
		v_val = ((y_val-y_val.mean())**2).sum()
		v_train = ((y_train-y_train.mean())**2).sum()
		
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
		
		la_b = Lasso(alpha=baseline_reg_lasso)
		la_b.fit(X_train,y_train)
		val_lasso_baseline.append(la_b.score(X_val,y_val))
		train_lasso_baseline.append(la_b.score(X_train,y_train))
	out_val_ridge.append(np.median(val_ridge)),out_val_RERFs.append(np.median(val_RERFs)),out_val_lasso.append(np.median(val_lasso)),out_train_ridge.append(np.median(train_ridge)),out_train_RERFs.append(np.median(train_RERFs)),out_train_lasso.append(np.median(train_lasso)) 
	out_val_lasso_baseline.append(np.median(val_lasso_baseline))
	out_train_lasso_baseline.append(np.median(train_lasso_baseline))

	
val_ridge,val_RERFs,val_lasso,train_ridge,train_RERFs,train_lasso =out_val_ridge,out_val_RERFs,out_val_lasso,out_train_ridge,out_train_RERFs,out_train_lasso
plt.figure()
mev_r,mev_R,mev_l=np.mean(val_ridge),np.mean(val_RERFs),np.mean(out_val_lasso_baseline)
mav_r,mav_R,mav_l=np.median(val_ridge),np.median(val_RERFs),np.median(out_val_lasso_baseline)
plt.semilogy(val_ridge,'b-',label='val_ridge(baseline)')
plt.semilogy(val_RERFs,'r-',label='val RERFs')
plt.semilogy(out_val_lasso_baseline,'g-',label='val lasso(baseline)')
plt.semilogy(train_RERFs,'r:',label='train_RERfs')
plt.semilogy(out_train_lasso_baseline,'g.',label='train_lasso(baseline)')
plt.semilogy(train_ridge,'b:',label='train_ridge')
plt.semilogy([0,num_cv],[mav_l,mav_l],'g-.',label='median val lasso(baseline)')
plt.semilogy([0,num_cv],[mav_R,mav_R],'m--',label='median val RERFs')
plt.semilogy([0,num_cv],[mav_r,mav_r],'c--',label='median val ridge')
plt.legend()
plt.savefig('compare.png',format='png',dpi=300,bbox_inches='tight')
#plt.show()
print('mean ridg',mev_r)
print('mean RERF',mev_R)
print('median ridg',mav_r)
print('median RERFs',mav_R)
print('reg lasso' ,reg_lasso)
print('param_forest',param_forest)
