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
print('RERFs lasso parameters : ',la.get_params(deep=True))

rf = data['random_forest'].item()['rf']
print('RERFs forest parameters : ',rf.get_params(deep=True))
y_pred = la.predict(X)+rf.predict(X)
acc_RERFs = 1-((y-y_pred)**2).sum()/((y-y.mean())**2).sum()
print('RERFs accu : ' ,acc_RERFs)

ridge = data['baseline_ridge'].item()['ridge']
print('baseline ridge parameters : ',ridge.get_params(deep=True))
y_pred_ridge = ridge.predict(X)
acc_ridge = ridge.score(X,y)
print('baseline ridge accu : ' ,acc_ridge)

lasso = data['baseline_lasso'].item()['lasso']
print('baseline lasso parameters : ',lasso.get_params(deep=True))
y_pred_lasso=lasso.predict(X)
acc_lasso = lasso.score(X,y)
print('baseline lasso accu : ' ,acc_lasso)

idx= np.argsort(y)

plt.figure()
data = np.load('regression_data1.npz')
y_ = data['y']
X_=data['X']
idx_ = np.argsort(y_)
plt.plot(4*np.arange(y_.shape[0]),y_[idx_],label='initial data')
plt.plot(y[idx],label='true data')
plt.legend()
plt.savefig('initial_value_and_test_value.png')
plt.show()

idx_closest = [np.argmin((y[idx]-yk)**2) for yk in np.sort(y_)]



mse_RERFs=(1/num_data)*(y-y_pred)**2
m_RERFs = np.median(mse_RERFs)
mse_ridge =((y-y_pred_ridge)**2)/num_data
m_ridge=np.median(mse_ridge)
mse_lasso =((y-y_pred_lasso)**2)/num_data
m_lasso=np.median(mse_lasso)
plt.figure()
plt.plot(mse_RERFs[idx],'r,',label='mse RERFs')
plt.plot(mse_ridge[idx],'b,',label='mse ridge')
plt.plot([0,num_data],[m_RERFs,m_RERFs],label='mean mse RERfs'+str(np.round(m_RERFs,3)))
plt.plot([0,num_data],[m_ridge,m_ridge],label='mean mse ridge'+str(np.round(m_ridge,3)))
plt.legend()
plt.savefig('error.png')
plt.show()

print('accuracy RERFs : ' , acc_RERFs)
plt.figure()
plt.title('accuracy RERFs: '+str(np.round(acc_RERFs,3)))
idx_RERFs=np.arange(num_data)[mse_RERFs[idx]<m_RERFs]
plt.plot(y[idx],label='true_value')
plt.plot(y_pred[idx],',',label='pred RERFs')
#plt.plot(idx_closest,y_pred[idx[idx_closest]],'k,',label='closest to train target') 
plt.plot(idx_RERFs,y_pred[idx[idx_RERFs]],',',label='better than median')
plt.plot(idx_closest,y[idx[idx_closest]],'k,',label='point closest to train target')
plt.savefig('pred_RERFsvs_true.png')
plt.show()

print('accuracy Ridge (baseline): ',acc_ridge)
plt.figure()
plt.title('accuracy ridge (baseline): '+str(np.round(acc_ridge,3)))
idx_ridge=np.arange(num_data)[mse_ridge[idx]<m_ridge]
plt.plot(y[idx],label='true_value')
plt.plot(y_pred_ridge[idx],',',label='pred ridge')
plt.plot(idx_ridge,y_pred_ridge[idx[idx_ridge]],',',label='better than median')
#plt.plot(idx_closest,y_pred_ridge[idx[idx_closest]],'k,',label='closest to train target') 
plt.plot(idx_closest,y[idx[idx_closest]],'k,',label='point closest to train target' )
plt.legend()
plt.savefig('pred_ridge_vs_true.png')
plt.show()

print('accuracy lasso (baseline) : ',acc_lasso)
plt.figure()
plt.title('accuracy lasso (baseline): '+str(np.round(acc_lasso,3)))
idx_lasso=np.arange(num_data)[mse_lasso[idx]<m_lasso]
plt.plot(y[idx],label='true_value')
plt.plot(y_pred_lasso[idx],',',label='pred lasso')
plt.plot(idx_lasso,y_pred_lasso[idx[idx_ridge]],',',label='better than median')
#plt.plot(idx_closest,y_pred_lasso[idx[idx_closest]],'k,',label='closest to train target') 
plt.plot(idx_closest,y[idx[idx_closest]],'k,',label='point closest to train target' )
plt.legend()
plt.savefig('pred_lasso_vs_true.png')
plt.show()

