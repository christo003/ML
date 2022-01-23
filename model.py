import numpy as np 
import sys 
from sklearn.ensemble import RandomForestRegressor
from lasso import lasso 
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

data = np.load('regression_data1.npz')
X,y = data['X'],data['y']
num_data,num_feature = X.shape
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

acc_train = 1- ((y-(la.predict(X)+rf.predict(X)))**2).sum()/((y-y.mean())**2).sum()

rows,columns = 5,10
fig,ax_array = plt.subplots(rows,columns,squeeze=False)
k=0
idx=np.argsort(y)
num_point=num_data
#a=0
#X=X[idx[a:a+num_plot]]
#y=y[idx[a:a+num_plot]]
for i,ax_row in enumerate(ax_array):
        for j,axes in enumerate(ax_row):
                I=X[:,k]
                #axes.set_xticks(I)
                #axes.set_yticks(y)
                xx=np.zeros((num_point,num_feature))
                xx[:,k]=I
                yy=rf.predict(xx)+ la.predict(xx)
                axes.set_title(str(k))#+' acc: '+str(np.round(acc_train,3)))
                axes.plot(I,yy,',')
                k=k+1
plt.savefig('affichage_fonction.png')
fig,ax_array = plt.subplots(rows,columns,squeeze=False)
k=0
for i,ax_row in enumerate(ax_array):
        for j,axes in enumerate(ax_row):
                xx=np.zeros((num_point,num_feature))
                I=X[idx,k]
                xx[idx,k]=I
                yy=rf.predict(xx)+ la.predict(xx)
                axes.set_title(str(k))
                axes.plot(((y[idx]-yy[idx])**2)/num_data,',')
                k=k+1
plt.savefig('error_par_feature.png')
plt.show()

