import numpy as np 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import sys
from lasso import lasso

data = np.load('test.npz')

X , y =data['X'],data['y']

num_data,num_feature=X.shape

data = np.load('model.npz',allow_pickle=True)

lasso = data['lasso'].item()

a,m_lasso ,std_lasso = lasso['alpha'],lasso['m'],lasso['std']

X=(X-m_lasso)/std_lasso

rf = data['random_forest'].item()['rf']

print(rf.get_params(deep=True))


y_pred = np.dot(X,a)+rf.predict(X)

acc = 1-((y-y_pred)**2).sum()/((y-y.mean())**2).sum()


print('accuracy : ' , acc)
plt.figure()
idx= np.argsort(y)
plt.plot(y[idx],label='target')
plt.plot(y_pred[idx],',',label='pred target')
plt.legend()
plt.savefig('pred_vs_true.png')
plt.show()

plt.figure()
data = np.load('regression_data1.npz')
y_ = data['y']
idx_ = np.argsort(y_)
plt.plot(y_[idx_],label='initial data')
plt.plot(y[idx],label='true data')
plt.legend()
plt.savefig('initial_value_and_test_value.png')
plt.show()
