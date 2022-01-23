import numpy as np 
import sys 
from sklearn.ensemble import RandomForestRegressor
from lasso import lasso 
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

data = np.load('regression_data1.npz')
X,y = data['X'],data['y']
num_data,num_feature_lasso = X.shape
parameters=np.load('parameters.npz',allow_pickle=True)
param_forest,param_lasso=parameters['RERFs_param_forest'],parameters['RERFs_param_lasso']
reg_ridge = parameters['reg_ridge']
reg_lasso = parameters['reg_lasso']
param_forest = param_forest.item()
#param_forest['criterion']='mse'#'squared_error'
la = Lasso(param_lasso)
la.fit(X,y)

rf = RandomForestRegressor(**param_forest)
print(rf.get_params())
y_f = y-la.predict(X)#np.dot(X,a)
rf.fit(X,y_f,np.ones(num_data))

ridge = Ridge(alpha=reg_ridge)
ridge.fit(X,y)



baseline_lasso=Lasso(alpha=reg_lasso)
baseline_lasso.fit(X,y)
print('RERFs_param_lasso',param_lasso)
print('RERFs_param_forest',param_forest)
print('baseline reg_lasso',reg_lasso)
print('baseline reg_ridge',reg_ridge)
np.savez('model.npz',lasso={'lasso':la},random_forest={'rf':rf},baseline_ridge={'ridge':ridge},baseline_lasso={'lasso':baseline_lasso })
