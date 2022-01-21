import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.ensemble import RandomForestRegressor

data = np.load('regression_data2.npz')
X,y = data['X'],data['y']
num_data,num_feature = X.shape

I = np.arange(X.shape[0])
np.random.shuffle(I)
rows,columns = 10,10
num_folder = rows*columns
num_val = int(num_data/num_folder)
max_samples=np.random.uniform(0.3,0.4,num_folder)
#fig,ax_array= plt.subplots(rows,columns,squeeze=False)
fig = plt.figure()
k,mse,out_bag,error_mse,error_bag = 0,sys.maxsize,sys.maxsize,[],[]
for i in range(rows):
	for j in range(columns):
		print('\t num_folder : ',k+1, ' / ' ,num_folder) 
		X_val,y_val = X[I[k*num_val:(k+1)*num_val]],y[I[k*num_val:(k+1)*num_val]]
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

		regr = RandomForestRegressor(oob_score=True,max_samples=max_samples[k],max_features='sqrt')#,ccp_alpha=ccp_alpha[k])
		regr.fit(X_train,y_train)
		y_pred = regr.predict(X_val)
		
		#print(regr.get_params(deep=True))
		
		#axes.set_title(str(k).format(i,j))
		error_mse.append(np.mean(np.abs(y_val-y_pred )))
		error_bag.append(np.abs(regr.oob_score_))
		plt.semilogy(max_samples[k],np.mean(np.abs(y_val-y_pred)),'or')
		if out_bag >np.abs(regr.oob_score_):
			out_bag = np.abs(regr.oob_score_)
			out_bag_indice=k
		if mse>np.mean(np.abs(y_val-y_pred)):
			mse = np.mean(np.abs(y_val-y_pred))
			mse_indice=k
		plt.semilogy(max_samples[k],np.abs(regr.oob_score_),'ob')
		#print(max_samples[k])
		#print(regr.score(X_val,y_val))
		#print(regr.decision_path(X_val))
		k=k+1
plt.semilogy(max_samples[k-1],np.abs(regr.oob_score_),'ob',label='out of bag error')
plt.semilogy(max_samples[k-1],np.mean(np.abs(y_val-y_pred)),'or',label = 'mse pred')
plt.semilogy(max_samples[out_bag_indice],out_bag,'om',label='best out of bag')
plt.semilogy(max_samples[mse_indice],mse,'og',label='best mse')
plt.semilogy([0,1],[np.median(error_mse),np.median(error_mse)])
plt.semilogy([0,1],[np.median(error_bag),np.median(error_bag)])
plt.title('cross vall on max_samples parameters')
plt.legend()
plt.show()

	
