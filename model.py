import numpy as np 
import sys 
from sklearn.ensemble import RandomForestRegressor
from lasso import lasso 

data = np.load('regression_data1.npz')
X,y = data['X'],data['y']
num_data,num_feature_lasso = X.shape
parameters=np.load('parameters.npz',allow_pickle=True)
alpha,param_forest,m_lasso,std_lasso,reg_lasso=parameters['model_lasso'],parameters['param_forest'],parameters['m_lasso'],parameters['std_lasso'],parameters['reg_lasso']

m,std = np.mean(X,0).reshape((1,num_feature_lasso)),np.std(X,0).reshape((1,num_feature_lasso))
X=(X-m)/std
a,_,_,_=lasso(X,y,np.zeros(num_feature_lasso),reg_lasso)


rf = RandomForestRegressor(**param_forest.item())

y_f = y-np.dot(X,a)
X_f=X


rf.fit(X_f,y_f,np.ones(num_data))

print('reg_lasso',reg_lasso)
print(rf.feature_importances_)
np.savez('model.npz',lasso={'alpha':a,'m':m,'std':std},random_forest={'rf':rf} )
