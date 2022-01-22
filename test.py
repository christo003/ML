import numpy as np 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import sys
from lasso import lasso
from sklearn.linear_model import Lasso
data = np.load('test.npz')
X , y =data['X'],data['y']
num_data,num_feature=X.shape
data = np.load('model.npz',allow_pickle=True)
la = data['lasso'].item()['lasso']
#a,m_lasso ,std_lasso = lasso['alpha'],lasso['m'],lasso['std']
#X=(X-m_lasso)/std_lasso
rf = data['random_forest'].item()['rf']
ridge = data['ridge'].item()['ridge']
print(rf.get_params(deep=True))
y_pred = la.predict(X)+rf.predict(X)
acc_RERFs = 1-((y-y_pred)**2).sum()/((y-y.mean())**2).sum()
print('accuracy RERFs : ' , acc_RERFs)

y_pred_ridge = ridge.predict(X)
acc_ridge = 1-((y-y_pred_ridge)**2).sum()/((y-y.mean())**2).sum()
print('accuracy Ridge : ',acc_ridge)
plt.figure()
idx= np.argsort(y)
plt.plot(y[idx],label='true_value')
plt.plot(y_pred[idx],',',label='pred RERFs')
plt.plot(y_pred_ridge[idx],',',label='pred ridge')
plt.legend()
plt.savefig('pred_vs_true.png')
plt.show()

plt.figure()
plt.plot(np.abs(y[idx]-y_pred[idx]),'r,',label='absolut error RERFs')
plt.plot(np.abs(y[idx]-y_pred_ridge[idx]),'b,',label='absolut error ridge')
plt.legend()
plt.savefig('error.png')
plt.show()

plt.figure()
data = np.load('regression_data1.npz')
y_ = data['y']
idx_ = np.argsort(y_)
plt.plot(4*np.arange(y_.shape[0]),y_[idx_],label='initial data')
plt.plot(y[idx],label='true data')
plt.legend()
plt.savefig('initial_value_and_test_value.png')
plt.show()
