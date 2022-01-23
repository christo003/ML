import numpy as np 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import sys
from lasso import lasso
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

data = np.load('test.npz')
X , y =data['X'],data['y']
num_data,num_feature=X.shape
data = np.load('model.npz',allow_pickle=True)

la = data['lasso'].item()['lasso']
print('RERFs lasso parameters : \n',la.get_params(deep=True))
rf = data['random_forest'].item()['rf']
print('\nRERFs forest parameters : \n',rf.get_params(deep=True))
y_pred = la.predict(X)+rf.predict(X)
acc_RERFs = 1-((y-y_pred)**2).sum()/((y-y.mean())**2).sum()
print('\nRERFs accu : ' ,acc_RERFs)

ridge = data['baseline_ridge'].item()['ridge']
print('baseline ridge parameters : ',ridge.get_params(deep=True))
y_pred_ridge = ridge.predict(X)
acc_ridge = ridge.score(X,y)
print('baseline ridge accu : ' ,acc_ridge)

la_ = data['baseline_lasso'].item()['lasso']
print('baseline lasso parameters : ',la_.get_params(deep=True))
y_pred_lasso=la_.predict(X)
acc_lasso = la_.score(X,y)
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
#plt.show()

idx_closest = [np.argmin((y[idx]-yk)**2) for yk in np.sort(y_)]



mse_RERFs=(1/num_data)*(y-y_pred)**2
m_RERFs = np.median(mse_RERFs)
mse_ridge =((y-y_pred_ridge)**2)/num_data
m_ridge=np.median(mse_ridge)
mse_lasso =((y-y_pred_lasso)**2)/num_data
m_lasso=np.median(mse_lasso)
plt.figure()
#plt.ylim(10**(-3),10**(-2))
plt.semilogy(mse_RERFs[idx],'r,',label='mse RERFs')
plt.semilogy(mse_ridge[idx],'b,',label='mse ridge')
plt.semilogy([0,num_data],[m_RERFs,m_RERFs],label='mean mse RERfs'+str(np.round(m_RERFs,3)))
plt.semilogy([0,num_data],[m_ridge,m_ridge],label='mean mse ridge'+str(np.round(m_ridge,3)))
plt.legend()
plt.savefig('error.png')
#plt.show()

print('\naccuracy RERFs : ' , acc_RERFs)
plt.figure()
plt.title('accuracy RERFs: '+str(np.round(acc_RERFs,3)))
idx_RERFs=np.arange(num_data)[mse_RERFs[idx]<m_RERFs]
plt.plot(y[idx],label='true_value')
plt.plot(y_pred[idx],',',label='pred RERFs')
#plt.plot(idx_closest,y_pred[idx[idx_closest]],'k,',label='closest to train target') 
plt.plot(idx_RERFs,y_pred[idx[idx_RERFs]],',',label='better than median')
plt.plot(idx_closest,y[idx[idx_closest]],'k,',label='point closest to train target')
plt.savefig('pred_RERFsvs_true.png')
#plt.show()

print('\naccuracy Ridge (baseline): ',acc_ridge)
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
#plt.show()

print('\naccuracy lasso (baseline) : ',acc_lasso)
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
#plt.show()
print('\nRERFs : linear coef find with lasso \n',np.arange(num_feature)[la.coef_!=0])
print('RERFs: non linear coef find with lasso \n',np.arange(num_feature)[la.coef_==0])
print('RERFs :feature importance linear \n',np.argsort(np.abs(np.abs(la.coef_)-np.max(np.abs(la.coef_)))))
print('RERFs :feature importance non linear\n',np.argsort(np.abs(np.abs(rf.feature_importances_)-np.max(np.abs(rf.feature_importances_)))))
print('\nridge (baseline):feature importance linear \n',np.argsort(np.abs(np.abs(ridge.coef_)-np.max(np.abs(ridge.coef_)))))
print('\nlasso(baseline) :feature importance linear ',np.argsort(np.abs(np.abs(la_.coef_)-np.max(np.abs(la_.coef_)))))
print('\nlasso (baseline) : linear coef find with lasso \n',np.arange(num_feature)[la_.coef_!=0])
print('lasso (baseline) : non linear coef find with lasso\n',np.arange(num_feature)[la_.coef_==0])

#la_RERFs_true = Lasso(**la.get_params(deep=True))
#la_RERFs_true.fit(X,y)

#rf_RERFs_true = RandomForestRegressor(**rf.get_params(deep=True))
#rf_RERFs_true.fit(X,y-la_RERFs_true.predict(X))

data=np.load('my_model_train_on_test.npz',allow_pickle=True)
rf_RERFs_true=data['random_forest'].item()['random_forest']
la_RERFs_true=data['lasso'].item()['lasso']

#np.savez('my_model_train_on_test.npz',random_forest={'random_forest':rf_RERFs_true},lasso={'lasso':la_RERFs_true})

y_pred_RERFs_true = la_RERFs_true.predict(X)+rf_RERFs_true.predict(X)
acc_test_RERFs = 1-((y-y_pred_RERFs_true)**2).sum()/((y-y.mean())**2).sum()


print('arruracy my model with test as train ',acc_test_RERFs)


print('\nRERFs true : linear coef find with lasso \n',np.arange(num_feature)[la_RERFs_true.coef_!=0])
print('RERFs true: non linear coef find with lasso\n',np.arange(num_feature)[la_RERFs_true.coef_==0])
print('RERFs true:feature importance linear \n',np.argsort(np.abs(np.abs(la_RERFs_true.coef_)-np.max(np.abs(la_RERFs_true.coef_)))))
print('RERFs true:feature importance non linear\n',np.argsort(np.abs(np.abs(rf_RERFs_true.feature_importances_)-np.max(np.abs(rf_RERFs_true.feature_importances_)))))
print()


#la_true = Lasso(alpha=0.1)
#la_true.fit(X,y)
#rf_true = RandomForestRegressor()
#rf_true.fit(X,y-la_true.predict(X))

data=np.load('best_model_train_on_test.npz',allow_pickle=True)
rf_true=data['random_forest'].item()['random_forest']
la_true=data['lasso'].item()['lasso']
#np.savez('best_model_train_on_test.npz',random_forest={'random_forest':rf_true},lasso={'lasso':la_true})


y_pred_true = rf_true.predict(X)+la_true.predict(X)
#y_pred_true = rf_true.predict(X)+y_lasso_true
#y_pred_true = rf_true.predict(X)
acc_true = 1-((y-y_pred_true)**2).sum()/((y-y.mean())**2).sum()
print('accuracy standard model with test data as train',acc_true)

#print('\ntrue : linear coef find with lasso \n',np.arange(num_feature)[a!=0])
#print('true: non linear coef find with lasso\n',np.arange(num_feature)[a==0])
#print('true:feature importance linear \n',np.argsort(np.abs(np.abs(a)-np.max(np.abs(a)))))
print('true:feature importance non linear\n',np.argsort(np.abs(np.abs(rf_true.feature_importances_)-np.max(np.abs(rf_true.feature_importances_)))))
print('\ntrue : linear coef find with lasso \n',np.arange(num_feature)[la_true.coef_!=0])
print('true: non linear coef find with lasso\n',np.arange(num_feature)[la_true.coef_==0])
print('true:feature importance linear \n',np.argsort(np.abs(np.abs(la_true.coef_)-np.max(np.abs(la_true.coef_)))))
print('true:feature importance non linear\n',np.argsort(np.abs(np.abs(rf_true.feature_importances_)-np.max(np.abs(rf_true.feature_importances_)))))
print()

plt.figure()
a,b,c,d,e=np.abs(la_.coef_),np.abs(ridge.coef_),np.abs(la.coef_),np.abs(la_RERFs_true.coef_),np.abs(la_true.coef_)
plt.title('linear importance')
plt.plot(a/np.max(a),'<',label='baseline lasso')
plt.plot(b/np.max(b),'>',label='baseline ridge')
plt.plot(c/np.max(c),'o',label='RERFs lin')
plt.plot(d/np.max(d),'x',label='RERFs test lin')
plt.plot(e/np.max(e),'v',label='RERFs test opti lin')
plt.grid()#axis='x')
plt.xticks(np.linspace(0,49,50))
plt.legend()
plt.title('test lin. feature importance (when 1 the model say its important)')
plt.savefig('test_linear_feature_importance.png')

plt.figure()
a,b,c,d,e=np.abs(la_.coef_),np.abs(ridge.coef_),np.abs(la.coef_),np.abs(la_RERFs_true.coef_),np.abs(la_true.coef_)
I=np.arange(5,50,1)
plt.title('linear importance log scale feature 5 to 50')
plt.semilogy(I,(a/np.max(a))[5:],'<',label='baseline lasso')
plt.semilogy(I,(b/np.max(b))[5:],'>',label='baseline ridge')
plt.semilogy(I,(c/np.max(c))[5:],'o',label='RERFs lin')
plt.semilogy(I,(d/np.max(d))[5:],'x',label='RERFs test lin')
plt.semilogy(I,(e/np.max(e))[5:],'v',label='RERFs test opti lin')
plt.grid(axis='x')
plt.xticks(np.linspace(5,49,45))
plt.legend()
plt.title('test lin. feature importance (semilogy scale of feature 5 to 50)')
plt.savefig('test_linear_feature_importance2.png')



plt.figure()
plt.title('non lin importance')
a,b,c=rf.feature_importances_,rf_RERFs_true.feature_importances_,rf_true.feature_importances_
plt.semilogy(a/np.max(a),'o',label='RERFs non lin')
plt.semilogy(b/np.max(b),'x',label='RERFs test non lin')
plt.semilogy(c/np.max(c),'v',label='RERFs test opti non lin')
plt.xticks(np.linspace(0,49,50))
plt.grid(axis='x')
plt.legend()
plt.title('test non lin. feature importance (when 1 the model say its important)')
plt.savefig('test_non_linear_feature_importance.png')

rows,columns=5,10
fig,ax_array = plt.subplots(rows,columns,squeeze=False)
k=0
idx=np.argsort(y)
num_point=num_data
for i,ax_row in enumerate(ax_array):
        for j,axes in enumerate(ax_row):
                I=X[:,k]
                #axes.set_xticks(I)
                #axes.set_yticks(y)
                xx=np.zeros((num_point,num_feature))
                xx[:,k]=I
                yy=rf.predict(xx)+ la.predict(xx)
                axes.plot(I,yy,',',label='model train')
                yy=rf_true.predict(xx)+ la_true.predict(xx)
                axes.plot(I,yy,',',label='model opt test')
                yy=rf_RERFs_true.predict(xx)+ la_RERFs_true.predict(xx)
                axes.plot(I,yy,',',label='model test')
                axes.get_xaxis().set_visible(False)
                axes.get_yaxis().set_visible(False)

                k=k+1

fig.suptitle('test : fonction feature (when other is set to zeros)')
plt.legend()
plt.savefig('test_affichage_fonction_test.png')
fig,ax_array = plt.subplots(rows,columns,squeeze=False)
k=0
for i,ax_row in enumerate(ax_array):
        for j,axes in enumerate(ax_row):
                xx=np.zeros((num_point,num_feature))
                I=X[idx,k]
                xx[:,k]=I
                yy=rf.predict(xx)+ la.predict(xx)
                axes.set_ylim(10**(-6),10**(-2))
                axes.semilogy(((y[idx]-yy[idx])**2)/num_data,',',label='model train')
                yy=rf_true.predict(xx)+ la_true.predict(xx)
                axes.semilogy(((y[idx]-yy[idx])**2)/num_data,',',label='model opt test')
                yy=rf_RERFs_true.predict(xx)+ la_RERFs_true.predict(xx)
                axes.semilogy(((y[idx]-yy[idx])**2)/num_data,',',label='model test')
                axes.get_xaxis().set_visible(False)
                axes.get_yaxis().set_visible(False)
                k=k+1
fig.suptitle('test : error / feature')
plt.legend()
plt.savefig('test_error_par_feature.png')
plt.show()



	
	
			
		
