import numpy as np 
import sys
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from lasso import lasso

data=np.load('regression_data1.npz')
y,X= data['y'],data['X']
num_data,num_feature=X.shape


num_lasso = 100
num_tree=5

num_val_lasso=int(num_data/num_lasso)
num_val_tree=int(num_data/num_tree)


I_tree=[np.arange(num_data) for i in range(num_tree)]


for i in range(num_tree):
	np.random.shuffle(I_tree[i])

I_lasso=[np.arange(num_data) for i in range(num_lasso)]


out_n_estimators,out_max_features,out_max_depth,out_max_depth,out_min_samples_split,out_min_samples_leaf,out_bootstrap, out_ccp_alphas = [],[],[],[],[],[],[],[]

n_estimators=[int(x) for x in np.linspace(2000,100,10)]
max_features = ['auto', 'sqrt','log2']
max_depth=[int(x) for x in np.linspace(110,10,11)]
#max_depth.append(None)
min_samples_split = [2,5,10]
min_samples_leaf = [3,4,10]
bootstrap = [True,False]
ccp_alphas = np.linspace(0,1,5)
random_grid = {'n_estimators':n_estimators,'max_features':max_features,'max_depth':max_depth,'min_samples_split':min_samples_split,'min_samples_leaf' : min_samples_leaf,'bootstrap':bootstrap}

best=[]
for j in range(num_tree):
	best_score=-sys.maxsize
	print('num_tree : ',j+1, '/ ' , num_tree)

	for i in range(num_lasso):
		np.random.shuffle(I_lasso[i])
	
	
	mse_cross,L,A=[],[],[]
	for k in range(num_lasso):
		I = I_lasso[k]
		print('\t num_lasso : ',k+1, ' / ', num_lasso)
		X_val = X[I[k*num_val_lasso:(k+1)*num_val_lasso]]
		y_val = y[I[k*num_val_lasso:(k+1)*num_val_lasso]]
		m_val = np.mean(X_val,0).reshape((1,num_feature))
		std_val = np.std(X_val,0).reshape((1,num_feature))

		#cross validation statement
		if k==0:
			X_train,y_train = X[I[(k+1)*num_val_lasso:]],y[I[(k+1)*num_val_lasso:]]
		elif k == num_lasso:
			X_train,y_train=X[I[0:(k-1)*num_val_lasso]],y[I[0:(k-1)*num_val_lasso]]
		else: 
			XA,yA,XB,yB = X[I[0:(k-1)*num_val_lasso]],y[I[0:(k-1)*num_val_lasso]],X[I[(k+1)*num_val_lasso:]],y[I[(k+1)*num_val_lasso:]]
			X_train,y_train = np.concatenate((XA,XB)),np.concatenate((yA,yB))
		nval = X_train.shape[0]

		m_train,std_train  = np.mean(X_train,0).reshape((1,num_feature)),np.std(X_train,0).reshape((1,num_feature))
		X_train = (X_train - m_train ) / std_train
		reg = np.max(np.dot(X_train.T,y_train))
		a,_,_,_ = lasso(X_train,y_train,np.zeros(num_feature),reg)
		X_val = (X_val - m_train) / std_train
		mse = np.mean((np.dot(X_val,a)-y_val)**2)
		mse_cross.append(mse),L.append(reg),A.append(a)
	
	I=I_tree[j]
	X_val = X[I[j*num_val_tree:(j+1)*num_val_tree]]
	y_val = y[I[j*num_val_tree:(j+1)*num_val_tree]]
	m_val = np.mean(X_val,0).reshape((1,num_feature))
	std_val = np.std(X_val,0).reshape((1,num_feature))

	#cross validation statement
	if j==0:
		X_train,y_train = X[I[(j+1)*num_val_tree:]],y[I[(j+1)*num_val_tree:]]
	elif j == num_tree:
		X_train,y_train=X[I[0:(j-1)*num_val_tree]],y[I[0:(j-1)*num_val_tree]]
	else: 
		XA,yA,XB,yB = X[I[0:(j-1)*num_val_tree]],y[I[0:(j-1)*num_val_tree]],X[I[(j+1)*num_val_tree:]],y[I[(j+1)*num_val_tree:]]
		X_train,y_train = np.concatenate((XA,XB)),np.concatenate((yA,yB))
	nval = X_train.shape[0]
	

	idx = np.argmin(mse_cross)
	reg,alpha = L[idx],A[idx]
	m_train,std_train=np.mean(X,0).reshape((1,num_feature)),np.std(X,0).reshape((1,num_feature))
	X_train=(X_train-m_train)/std_train
	X_val = (X_val-m_train)/std_train
	alpha,_,_,_ = lasso(X_train,y_train,alpha,reg)
	residual = y_train- np.dot(X_train,alpha)
	idx_no_zero = np.arange(num_feature)[alpha!=0]

	
	X_train=X_train[:,idx_no_zero]
	y_train = residual

	
	y_val = y_val-np.dot(X_val,alpha)
	X_val=X_val[:,idx_no_zero]

	rfs,val_acc,train_acc = [],[],[] 

	for n_estimator in n_estimators:
		print('\t\t n_estimator ',n_estimator, ' / ',n_estimators )
		for max_feature in max_features:
			print('\t\t  max_feature ',max_feature, ' / ',max_features )
			for depth in max_depth:
				print('\t\t   depth ',depth, ' / ',max_depth )
				for min_sample_split in min_samples_split:
					print('\t\t    min_samples_split ',min_sample_split, ' / ',min_samples_split )
					for min_sample_leaf in min_samples_leaf:
						print('\t\t     min_sample_leaf ',min_sample_leaf, ' / ',min_samples_leaf )
						for boot in bootstrap:
							print('\t\t      bootstrap ',boot, ' / ',bootstrap )
							for ccp_alpha in ccp_alphas:
								print('\t\t       ccp_alpha ',ccp_alpha, ' / ',ccp_alphas)
								rf = RandomForestRegressor(n_estimators=n_estimator,max_features=max_feature,max_depth=depth,min_samples_split=min_sample_split,min_samples_leaf=min_sample_leaf,bootstrap=boot)
								rf.fit(X_train,y_train)
								score=rf.score(X_val,y_val)
								if score>best_score:
									best_score=score
									best_rf = rf	
									best_train_acc=rf.score(X_train,y_train)
	rf = best_rf
	rfs.append(rf)
	val_acc.append(best_score)
	train_acc.append(best_train_acc)
	param = rf.get_params(deep=True)
	out_n_estimators.append(param['n_estimators'])
	out_max_features.append(param['max_features'])
	out_max_depth.append(param['max_depth'])
	out_min_samples_split.append(param['min_samples_split'])
	out_min_samples_leaf.append(param['min_samples_leaf'])
	out_bootstrap.append(param['bootstrap'])
	out_ccp_alphas.append(param['ccp_alpha'])
	best.append(rf)


print('out_n_estimators',out_n_estimators)
print('out_max_features',out_max_features)
print('out_max_depth',out_max_depth)
print('out_min_samples_split',out_min_samples_split)
print('out_min_samples_leaf',out_min_samples_leaf)
print('out_bootstrap',out_bootstrap)
print('out_ccp_alpha',out_ccp_alphas)

print('n_estimator',np.median(out_n_estimators))
auto,log2,sqrt=0,0,0
for i in out_max_features:
	if i=='auto':
		auto+=1
	if i=='log2':
		log2+=1
	if i=='sqrt':
		sqrt+=1

if (auto>=sqrt)&(auto>=log2):
	print('max_features auto')
if (auto<=sqrt)&(sqrt>=log2):
	print('max_features sqrt')
if (auto<=log2)&(sqrt<=log2):
	print('max_features log2')
out_max_depth = [sys.maxsize if i == None else i for i in out_max_depth]
print('max_depth',np.median(out_max_depth))
print('min_samples_split',np.median(out_min_samples_split))
print('min_samples_leaf',np.median(out_min_samples_leaf))


if np.sum(out_bootstrap)>num_tree/2:
	print('bootstrap True')
else:
	print('bootstrap False')

print('ccp_alpha',np.median(out_ccp_alphas))
print('train_acc',train_acc)
print('val_acc',val_acc)
