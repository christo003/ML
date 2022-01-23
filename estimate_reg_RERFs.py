import numpy as np 
import sys 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from lasso import lasso
from sklearn.linear_model import Lasso

data=np.load('regression_data1.npz')
y,X= data['y'],data['X']
num_data,num_feature=X.shape
max_reg = 100
logscale =lambda k: np.exp(np.log(1/(10*max_reg))+k*(np.log(max_reg)-np.log(1/(10*max_reg)))/(max_reg-1))

num_cv=5
num_lasso =10
num_ridge=10


#print(uu)
n_estimators=[100]
max_features = ['sqrt','log2']
num_max_samples=1	
num_max_depth=1
num_sample_split=1
num_min_samples_leaf=1
num_min_impurity_decrease=1

max_samples = list(np.unique(np.random.uniform(0.94,0.98,num_max_samples)))
max_samples.append(np.random.uniform(0.85,0.88))
max_samples.append(1)
print('max_samples',max_samples)
max_depth=list(np.unique(np.random.randint(3,14,num_max_depth)))
max_depth.append(17)
max_depth.append(29)
max_depth=list(np.unique(max_depth))
max_depth.append(None)
print('max_depth : ',max_depth)
min_samples_split =list(np.unique(np.random.randint(20,30,num_sample_split)))
min_samples_split.append(17)
min_samples_split.append(9)
min_samples_split=np.unique(min_samples_split)
print('min_samples_split',min_samples_split)
min_samples_leaf = list(np.unique(np.random.randint(25,35,num_min_samples_leaf)))
min_samples_leaf.append(15)
min_samples_leaf.append(7)
min_samples_leaf=np.unique(min_samples_leaf)
print('min_samples_leaf',min_samples_leaf)
min_impurity_decrease= list(np.unique(np.random.uniform(0.12,0.18,1)))
min_impurity_decrease.append(np.random.uniform(0.2,0.3,))
min_impurity_decrease.append(np.random.uniform(0.05,0.1))
min_impurity_decrease.append(0)
print('min_impurity_decrease',min_impurity_decrease)

RIDGE = [logscale(k) for k in  (90/(num_ridge)*(np.arange(0,num_ridge))).astype(int)+np.sort(np.random.randint(0,10,num_ridge))]
print('range ridge reg :',RIDGE)

LASSO = [logscale(k) for k in  (90/(num_lasso)*(np.arange(0,num_lasso))).astype(int)+np.sort(np.random.randint(0,10,num_lasso))]#np.unique(np.linspace(25,35,num_lasso).astype(int))]
print('range of lasso reg :', LASSO)

num_fold=int(num_data/num_cv)
I_all=[np.arange(num_data) for i in range(num_cv)]
for i in range(num_cv):
	np.random.shuffle(I_all[i])


out_n_estimators,out_max_features,out_max_depth,out_max_depth,out_min_samples_split,out_min_samples_leaf,out_max_samples, out_min_impurity_decrease = [],[],[],[],[],[],[],[]
out_lasso_best_reg=[]

out_val_ridge,out_val_lasso=[],[]




defaut_train,default_val=[],[]
out_lambda=[]
best=[]
rfs,val_acc,train_acc = [],[],[] 
out_ridge_train,out_ridge_val,out_alpha_ridge = [],[],[]
out_train_RERFs,out_val_RERFs=[],[]

parameters=[]
for j in range(num_cv):
	
#	max_samples = list(np.unique(np.random.uniform(0.8,1,num_max_samples)))
#	max_samples.append(1)
#	max_depth=np.unique(np.random.randint(17,40,num_max_depth))
#	min_samples_split =np.unique(np.random.randint(10,20,num_sample_split))
#	print('min_samples_split',min_samples_split)
#	print('max_depti' ,max_depth)
#	min_samples_leaf = np.unique(np.random.randint(5,16,num_min_samples_leaf))
#	print('min_samples_leaf',min_samples_leaf)
#	min_impurity_decrease= list(np.unique(np.random.uniform(0,0.2,num_min_impurity_decrease)))
#	min_impurity_decrease.append(0)
#	print('max_samples' ,max_samples)
	
	best_score=-sys.maxsize
	print('num_cv : ',j+1, '/ ' , num_cv)
	
	I = I_all[j]
	X_val = X[I[j*num_fold:(j+1)*num_fold]]
	y_val = y[I[j*num_fold:(j+1)*num_fold]]
	#cross validation statement
	if j==0:
		X_train,y_train = X[I[(j+1)*num_fold:]],y[I[(j+1)*num_fold:]]
	elif j == num_fold:
		X_train,y_train=X[I[0:(j-1)*num_fold]],y[I[0:(j-1)*num_fold]]
	else: 
		XA,yA,XB,yB = X[I[0:(j-1)*num_fold]],y[I[0:(j-1)*num_fold]],X[I[(j+1)*num_fold:]],y[I[(j+1)*num_fold:]]
		X_train,y_train = np.concatenate((XA,XB)),np.concatenate((yA,yB))
	nval = X_train.shape[0]

	m_train,std_train  = np.mean(X_train,0).reshape((1,num_feature)),np.std(X_train,0).reshape((1,num_feature))

	num_train,num_val = X_train.shape[0],X_val.shape[0]
	#X_train = (X_train - m_train ) / std_train
	
	#X_val = (X_val - m_train) / std_train
	
	best_ridge=-sys.maxsize
	
	for ridge in RIDGE:
		
		rr = Ridge(alpha=ridge)
		rr.fit(X_train,y_train)
		score_ridge = rr.score(X_val,y_val)
		if score_ridge>best_ridge:
			best_ridge = score_ridge
			train_ridge = rr.score(X_train,y_train)
			alpha_ridge = ridge
	out_ridge_train.append(train_ridge)
	out_ridge_val.append(best_ridge)
	print('best ridge on valisaton : ' ,best_ridge)
	out_alpha_ridge.append(alpha_ridge)

	
	alpha = np.zeros(num_feature)
	best_lasso_score=-sys.maxsize


	k=0
	for reg in  LASSO :
		print('\t num_lasso : ',k+1, ' / ', num_lasso)
		k=k+1
		
		print('\t\treg',reg)
		#reg = REG[k]


		la = Lasso(alpha = reg)
		la.fit(X_train,y_train)
		lasso_score = la.score(X_val,y_val)
		y_pred_val=la.predict(X_val)
		y_pred_train=la.predict(X_train)
		if best_lasso_score<lasso_score:
			best_lasso_score=lasso_score
			best_lasso_reg=reg
	
		residual= y_train-la.predict(X_train)
	
		#residual = (y_train- np.dot(X_train,alpha))

	


		best_forest_score=-sys.maxsize
		

		train_target_forest=residual

		weight = np.ones(X_train.shape[0])

		for n_estimator in n_estimators:
			print('\t\t n_estimator ',n_estimator, ' / ',n_estimators )
			for max_feature in max_features:
				#print('\t\t  max_feature ',max_feature, ' / ',max_features )
				for depth in max_depth:
					#print('\t\t   depth ',depth, ' / ',max_depth )
					for min_sample_split in min_samples_split:
						#print('\t\t    min_samples_split ',min_sample_split, ' / ',min_samples_split )
						for min_sample_leaf in min_samples_leaf:
							#print('\t\t     min_sample_leaf ',min_sample_leaf, ' / ',min_samples_leaf )
							for max_sample in max_samples:
								#print('\t\t      bootstrap ',boot, ' / ',bootstrap )
								for min_impurity_decr in min_impurity_decrease:
									#print('\t\t       min_impurity_decr ',min_impurity_decrease, ' / ',min_impurity_decrease)
									rf = RandomForestRegressor(n_estimators=n_estimator,max_features=max_feature,max_depth=depth,min_samples_split=min_sample_split,min_samples_leaf=min_sample_leaf,bootstrap=True,oob_score=True,max_samples=max_sample,criterion='squared_error',min_impurity_decrease=min_impurity_decr)
									rf.fit(X_train,train_target_forest,sample_weight=weight)
									score=rf.score(X_val,y_val)
									if score>best_forest_score:
										best_forest_score=score
										best_rf = rf	
										best_X_train_acc=rf.score(X_train,train_target_forest)

									y_pred = y_pred_val+rf.predict(X_val)
									u = ((y_val-y_pred)**2).sum()
									v = ((y_val-y_val.mean())**2).sum()
									score = 1-u/v

									if score>best_score:
										best_score=score
										best_RERFs={'lasso_reg':reg,'tree_reg':rf.get_params(deep=True)}
										y_pred = y_pred_train+rf.predict(X_train)
										u = ((train_target_forest-y_pred)**2).sum()
										v = ((train_target_forest-train_target_forest.mean())**2).sum()
										best_RERFs_train_acc = 1-u/v
										best_model = [la,rf]
		

		rf = RandomForestRegressor()
		rf.fit(X_train,train_target_forest)
		default_val.append(rf.score(X_val,y_val))
		defaut_train.append(rf.score(X_train,train_target_forest))
	print('\t best reg lasso on validatoin ', best_lasso_reg)
	out_val_lasso.append(lasso_score)
	out_lasso_best_reg.append(best_lasso_reg)
	la,rf=best_model
	param=rf.get_params(deep=True)
	best_model[0]=param
	best_model[1]=la.get_params()['alpha']
	rfs.append(best_model)
	val_acc.append(best_score)
	train_acc.append(best_RERFs_train_acc)
	parameters.append(param)
	out_n_estimators.append(param['n_estimators'])
	out_max_features.append(param['max_features'])
	out_min_samples_split.append(param['min_samples_split'])
	out_min_samples_leaf.append(param['min_samples_leaf'])
	out_max_samples.append(param['max_samples'])
	out_max_depth.append(param['max_depth'])
	out_min_impurity_decrease.append(param['min_impurity_decrease'])
	out_lambda.append(best_model[1])
	y_pred = la.predict(X_val)+rf.predict(X_val)
	u=((y_val-y_pred)**2).sum()
	v = ((y_val-y_val.mean())**2).sum()
	out_val_RERFs.append(1-u/v)
	
	print('\t parameters tree ',param)
	print('\t reg lasso', reg)
	y_pred = la.predict(X_train)+rf.predict(X_train)
	print('\t acc RERFs validation : ' ,1-u/v)
	u=((y_train-y_pred)**2).sum()
	v = ((y_train-y_train.mean())**2).sum()
	out_train_RERFs.append(1-u/v)
	
	


print('out_alpha_ridge',out_alpha_ridge)
print('out_ridge_train',out_ridge_train)

print('out_lambda',out_lambda)
print()
print('out_n_estimators',out_n_estimators)
print('out_max_features',out_max_features)
print('out_min_samples_split',out_min_samples_split)
print('out_min_samples_leaf',out_min_samples_leaf)
print('out_max_samples',out_max_samples)
print('out_min_impurity_decr',out_min_impurity_decrease)
print('out_max_deapth',out_max_depth)



idxi = np.argmin((out_val_RERFs-np.median(out_val_RERFs))**2)
idxir=np.argmin((out_ridge_val-np.median(out_ridge_val))**2)
idxil = np.argmin((out_val_lasso-np.median(out_val_lasso))**2)

print()
print('best reg lasso baseline' ,out_lasso_best_reg[idxil])
print()
print('reg ridge',out_alpha_ridge[idxir])

print()
print('lambda',out_lambda[idxi])
print()
auto,log2,sqrt=0,0,0
for i in out_max_features:
	if i=='auto':
		auto+=1
	if i=='log2':
		log2+=1
	if i=='sqrt':
		sqrt+=1
t=''
if (auto>=sqrt)&(auto>=log2):
	t=', max_features = auto'
if (auto<=sqrt)&(sqrt>=log2):
	t=', max_features = sqrt'
if (auto<=log2)&(sqrt<=log2):
	t=', max_features = log2'


print('n_estimator=',out_n_estimators[idxi],t,', max_depth = ',out_max_depth[idxi],', min_samples_split = ',out_min_samples_split[idxi],', min_samples_leaf = ',out_min_samples_leaf[idxi],', max_samples = ',out_max_samples[idxi],', min_impurity_decrease = ',out_min_impurity_decrease[idxi])
print()



#print('train_acc tree optimize',np.median(train_acc))
print('train_acc ridige optimizei',np.median(out_ridge_train))
print('train acc RERFs',out_train_RERFs)


#print('defaut val accu tree default'  ,np.median(default_val))
#print('val_acc tree optimize',np.median(val_acc))
print('val_acc ridge optimize ' ,np.max(out_ridge_val))
print('val acc RERFs', np.max(out_val_RERFs))
print('val acc RERFs',out_val_RERFs)
print('out_ridge_val',out_ridge_val)

rf,la=rfs[idxi]
np.savez('parameters.npz',reg_lasso=out_lasso_best_reg[idxil],RERFs_param_forest = rf,RERFs_param_lasso=la,reg_ridge = out_alpha_ridge[idxir])

