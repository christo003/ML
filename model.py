import numpy as np 
import sys 
from sklearn.ensemble import RandomForestRegressor
from lasso import lasso 
from sklearn.linear_model import Ridge

data = np.load('regression_data1.npz')
X,y = data['X'],data['y']
num_data,num_feature_lasso = X.shape
parameters=np.load('parameters.npz',allow_pickle=True)
alpha,param_forest,m_lasso,std_lasso,reg_lasso=parameters['model_lasso'],parameters['param_forest'],parameters['m_lasso'],parameters['std_lasso'],parameters['reg_lasso']
reg_ridge = parameters['reg_ridge']
param_forest = param_forest.item()
param_forest['criterion']='mse'#'squared_error'
m,std = np.mean(X,0).reshape((1,num_feature_lasso)),np.std(X,0).reshape((1,num_feature_lasso))
X=(X-m)/std
a,_,_,_=lasso(X,y,np.zeros(num_feature_lasso),reg_lasso)

rf = RandomForestRegressor(**param_forest)
print(rf.get_params())
y_f = y-np.dot(X,a)
rf.fit(X,y_f,np.ones(num_data))

ridge = Ridge(alpha=reg_ridge)
ridge.fit(X,y)

print('reg_lasso',reg_lasso)
print('reg_ridge',reg_ridge)
np.savez('model.npz',lasso={'alpha':a,'m':m,'std':std},random_forest={'rf':rf},ridge={'ridge':ridge} )
