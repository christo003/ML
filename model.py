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
param_forest,param_lasso=parameters['RERFs_param_forest'].item(),parameters['RERFs_param_lasso']

if (0<param_forest['max_samples'])&(param_forest['max_samples']<=1):
        param_forest['max_samples']=int(param_forest['max_samples']*num_data)


reg_ridge = parameters['reg_ridge']
reg_lasso = parameters['reg_lasso']
#param_forest['criterion']='mse'#'squared_error'
la = Lasso(alpha=param_lasso)
la.fit(X,y)

rf = RandomForestRegressor(**param_forest)
y_f = y-la.predict(X)#np.dot(X,a)
rf.fit(X,y_f,np.ones(num_data))

ridge = Ridge(alpha=reg_ridge)
ridge.fit(X,y)



baseline_lasso=Lasso(alpha=reg_lasso)
baseline_lasso.fit(X,y)
print('RERFs_param_lasso',param_lasso)
print('\nRERFs_param_forest\n',param_forest)
print('\nbaseline reg_lasso',reg_lasso)
print('\nbaseline reg_ridge',reg_ridge)
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
                #axes.set_title(str(k))#+' acc: '+str(np.round(acc_train,3)))
                axes.plot(I,yy,',')
                axes.get_xaxis().set_visible(False)
                axes.get_yaxis().set_visible(False)
                k=k+1

fig.suptitle('train: fonction model in respect of x /feature (when other to 0)')
plt.savefig('train_affichage_fonction.png')
fig,ax_array = plt.subplots(rows,columns,squeeze=False)
k=0
for i,ax_row in enumerate(ax_array):
        for j,axes in enumerate(ax_row):
                xx=np.zeros((num_point,num_feature))
                I=X[idx,k]
                xx[idx,k]=I
                yy=rf.predict(xx)+ la.predict(xx)
                #axes.set_title(str(k))
                axes.semilogy(((y[idx]-yy[idx])**2)/num_data,',')
                axes.get_xaxis().set_visible(False)
                axes.get_yaxis().set_visible(False)
                k=k+1
fig.suptitle('train : err / feature on training')
plt.savefig('train_error_par_feature.png')

plt.figure()
a,b,c=np.abs(baseline_lasso.coef_),np.abs(ridge.coef_),np.abs(la.coef_)
plt.title('train : linear importance')
plt.plot(a/np.max(a),'<',label='baseline lasso')
plt.plot(b/np.max(b),'>',label='baseline ridge')
plt.plot(c/np.max(c),'o',label='RERFs lin')
plt.grid()#axis='x')
plt.xticks(np.linspace(0,49,50))
plt.legend()
plt.title('train :lin. feature importance (when 1 the model say its important)')
plt.savefig('train_linear_feature_importance.png')

plt.figure()
a,b,c=np.abs(baseline_lasso.coef_),np.abs(ridge.coef_),np.abs(la.coef_)
I=np.arange(5,50,1)
plt.title('train:linear importance log scale feature 5 to 50') 
plt.semilogy(I,(a/np.max(a))[5:],'<',label='baseline lasso') 
plt.semilogy(I,(b/np.max(b))[5:],'>',label='baseline ridge') 
plt.semilogy(I,(c/np.max(c))[5:],'o',label='RERFs lin')
plt.grid(axis='x')
plt.xticks(np.linspace(5,49,45))
plt.legend()

plt.savefig('train_linear_feature_importance2.png')

plt.figure()
plt.title('train : non lin importance')
a=rf.feature_importances_
plt.semilogy(a/np.max(a),'o',label='RERFs non lin')
plt.xticks(np.linspace(0,49,50))
plt.grid(axis='x')
plt.legend()
plt.title('non lin. feature importance (when 1 the model say its important)')
plt.savefig('train_non_linear_feature_importance.png')
plt.show()
