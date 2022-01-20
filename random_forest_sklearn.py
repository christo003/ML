import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

data = np.load('regression_data2.npz')
X,y = data['X'],data['y']
num_data,num_feature = X.shape

I = np.arange(X.shape[0])
np.random.shuffle(I)
rows,columns = 3,3 
num_folder = rows*columns
num_val = int(num_data/num_folder)
regr = RandomForestRegressor()
columns=3
rows=3
fig,ax_array= plt.subplots(rows,columns,squeeze=False)
k=0
for i,ax_row in enumerate(ax_array):
	for j ,axes in enumerate(ax_row):
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
		
		regr.fit(X_train,y_train)
		y_pred = regr.predict(X_val)
		
		print(regr.get_params(deep=True))
		
		axes.set_title(str(k).format(i,j))
		axes.plot((y_val-y_pred)**2,'or')
		print(regr.score(X_val,y_val))
		print(regr.decision_path(X_val))
		k=k+1
plt.show()

	
