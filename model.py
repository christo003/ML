import numpy as np 
import sys 
from sklearn.ensemble import RandomForestRegressor
from lasso import lasso 

data = np.load('regression_data1.npz')
X,y = data['X'],data['y']
num_data,num_feature_lasso = X.shape
parameters=np.load('parameters.npz',allow_pickle=True)
alpha,param_forest,idx_no_zero,m_lasso,std_lasso,m_forest,std_forest,reg_lasso=parameters['model_lasso'],parameters['model_forest'],parameters['idx_no_zero'],parameters['m_lasso'],parameters['std_lasso'],parameters['m_forest'],parameters['std_forest'],parameters['reg_lasso']

m,std = np.mean(X,0).reshape((1,num_feature_lasso)),np.std(X,0).reshape((1,num_feature_lasso))
X=(X-m)/std
a,_,_,_=lasso(X,y,np.zeros(num_feature_lasso),reg_lasso)

print(param_forest)
rf = RandomForestRegressor(param_forest)

idx_no_zero= np.arange(num_feature_lasso)[a!=0]

print(idx_no_zero)

num_feature_forest = len(idx_no_zero)
y_f = y-np.dot(X,a)
X_f=X[:,idx_no_zero]

m_f,std_f = np.mean(X_f,0).reshape((1,num_feature_forest)),np.std(X_f,0).reshape((1,num_feature_forest))

X_f = (X_f-m_f)/std_f

rf.fit(X_f,y_f)

np.savez('model.npz',lasso={'alpha':a,'m':m,'std':std},rf={'rf':rf,'m':m_f,'std':std_f} )
