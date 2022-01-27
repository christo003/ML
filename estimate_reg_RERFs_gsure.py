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
logscale =lambda k,m: np.exp(np.log(1/(10*m))+k*(np.log(m)-np.log(1/(10*m)))/(m-1))

num_cv=4
num_lasso,num_ridge =4 , 4
n_estimators=[100]
max_features = ['auto']
num_max_samples=1	
num_max_depth=1
num_sample_split=3
num_min_samples_leaf=1
num_min_impurity_decrease=3
num_ccp_alpha=3

if num_max_samples>1:
	max_samples = list(1- np.array([logscale(k,3*num_max_samples)/(3*num_max_samples) for k in range(num_max_samples,2*num_max_samples,1)]))
else:
	max_samples=[]
max_samples.append(1)

print('\nmax_samples',max_samples)
if num_max_depth>1:
	max_depth =[int(l) for l in np.linspace(40,5,num_max_depth-1)]
else :
	max_depth = []
max_depth.append(None)
print('max_depth : ',max_depth)
if num_sample_split>1:
	min_samples_split =list(np.arange(4,4+(num_sample_split-1)**2,num_sample_split))
else:
	min_samples_split=[]
min_samples_split.append(2)
min_samples_split=np.unique(min_samples_split)
print('min_samples_split',min_samples_split)
if num_min_samples_leaf>1:

	min_samples_leaf =list(np.arange(3,3+(num_min_samples_leaf-1)**2,num_min_samples_leaf))
else : 
	min_samples_leaf =[]
min_samples_leaf.append(1)
#min_samples_leaf=np.unique(min_samples_leaf)
print('min_samples_leaf',min_samples_leaf)
if num_min_impurity_decrease>1:
	min_impurity_decrease= [logscale(k,3*num_min_impurity_decrease)/(3*num_min_impurity_decrease) for k in range(num_min_impurity_decrease,2*num_min_impurity_decrease,1)]
else : 
	min_impurity_decrease=[]
min_impurity_decrease.append(0)

print('min_impurity_decrease',min_impurity_decrease)

if num_ccp_alpha>1:
	ccp_alphas = [logscale(k,3*num_ccp_alpha)/(3*num_ccp_alpha) for k in range(num_ccp_alpha,2*num_ccp_alpha,1)]
else:
	ccp_alphas = []
ccp_alphas.append(0)

print('ccp_alpha : ' ,ccp_alphas)
if num_ridge>1:	
	RIDGE = [logscale(k,2*num_ridge)/(2*num_ridge) for k in range(num_ridge,2*num_ridge-1,1)]
else : 
	RIDGE=[]
RIDGE.append(0.1)
RIDGE=np.unique(RIDGE)
print('range ridge reg :',RIDGE)
if num_lasso>1:
	LASSO = [logscale(k,1000) for k in  429+np.arange(num_lasso-1)-int((num_lasso-1)/2)]
else:
	LASSO=[]
LASSO.append(.1)
LASSO = np.unique(LASSO)
num_lasso=len(LASSO)
print('range of lasso reg :', LASSO)

num_fold=int(num_data/num_cv)
I_all=[np.arange(num_data) for i in range(num_cv)]
for i in range(num_cv):
	np.random.shuffle(I_all[i])


out_n_estimators,out_max_features,out_max_depth,out_max_depth,out_min_samples_split,out_min_samples_leaf,out_max_samples, out_min_impurity_decrease = [],[],[],[],[],[],[],[]
out_lasso_best_reg=[]
out_ccp_alpha=[]
out_val_ridge,out_val_lasso=[],[]
defaut_train,default_val=[],[]
out_lambda=[]
best=[]
rfs,val_acc,train_acc = [],[],[] 
out_ridge_train,out_ridge_val,out_alpha_ridge = [],[],[]
out_train_RERFs,out_val_RERFs=[],[]

parameters=[]
for j in range(num_cv):
	best_score=-sys.maxsize
	print('\nnum_cv : ',j+1, '/ ' , num_cv)
	I = I_all[j]
	X_val = X[I[j*num_fold:(j+1)*num_fold]]
	y_val = y[I[j*num_fold:(j+1)*num_fold]]
	#cross validation statement
	if j==0:
		X_train,y_train = X[I[(j+1)*num_fold:]],y[I[(j+1)*num_fold:]]
	elif j == num_fold:
		X_train,y_train=X[I[0:(j-1)*num_fold]],y[I[0:(j-1)*num_fold]]
	else: 
		XA,yA,XB,yB = X[I[0:j*num_fold]],y[I[0:j*num_fold]],X[I[(j+1)*num_fold:]],y[I[(j+1)*num_fold:]]
		X_train,y_train = np.concatenate((XA,XB)),np.concatenate((yA,yB))
	m_train,std_train  = np.mean(X_train,0).reshape((1,num_feature)),np.std(X_train,0).reshape((1,num_feature))
	num_train,num_val = X_train.shape[0],X_val.shape[0]
	best_ridge=-sys.maxsize
	for ridge in RIDGE:
		
		rr = Ridge(alpha=ridge)
		rr.fit(X_train,y_train)
		score__ridge = rr.score(X_val,y_val)
		if score_ridge>best_ridge:
			best_ridge = score_ridge
			train_ridge = rr.score(X_train,y_train)
			alpha_ridge = ridge
	out_ridge_train.append(train_ridge)
	out_ridge_val.append(best_ridge)
	
	out_alpha_ridge.append(alpha_ridge)

	
	best_lasso_score=-sys.maxsize


	k=0
	for reg in  LASSO :
		print('\t num_lasso : ',k+1, ' / ', num_lasso, '  reg : ', reg)
		k=k+1
		
		la = Lasso(alpha = reg)
		la.fit(X_train,y_train)
		lasso_score = la.score(X_val,y_val)
		y_pred_val=la.predict(X_val)
		y_pred_train=la.predict(X_train)
		if best_lasso_score<lasso_score:
			best_lasso_score=lasso_score
			best_lasso_reg=reg
	
		residual= y_train-la.predict(X_train)
		train_target_forest=residual
		weight = np.ones(X_train.shape[0])/num_train
		for n_estimator in n_estimators:
			#print('\t\t n_estimator ',n_estimator, ' / ',n_estimators )
			for ccp_alpha in ccp_alphas:
				print('\t\t  ccp_alpha',ccp_alpha, ' / ',ccp_alphas )
				for max_feature in max_features:	
					#print('\t\t   max_feature ',max_feature, ' / ',max_features )
					for depth in max_depth:
						#print('\t\t    depth ',depth, ' / ',max_depth )
						for min_sample_split in min_samples_split:
							#print('\t\t     min_samples_split ',min_sample_split, ' / ',min_samples_split )
							for min_sample_leaf in min_samples_leaf:
								#print('\t\t      min_sample_leaf ',min_sample_leaf, ' / ',min_samples_leaf )
								for max_sample in max_samples:
									#print('\t\t       max_samples ',max_sample, ' / ',max_samples )
									for min_impurity_decr in min_impurity_decrease:
										#print('\t\t        min_impurity_decr ',min_impurity_decrease, ' / ',min_impurity_decr)
										rf = RandomForestRegressor(n_estimators=n_estimator,max_features=max_feature,max_depth=depth,min_samples_split=min_sample_split,min_samples_leaf=min_sample_leaf,bootstrap=True,oob_score=True,max_samples=int(num_train*max_sample),min_impurity_decrease=min_impurity_decr,ccp_alpha=ccp_alpha)
										rf.fit(X_train,train_target_forest,sample_weight=weight)

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
		
		

	print('\n\nridge acc : ',best_ridge,' with reg : ',alpha_ridge)
	print('Lasso acc : ',best_lasso_score,' with reg : ',best_lasso_reg)
	
	
	out_val_lasso.append(lasso_score),out_lasso_best_reg.append(best_lasso_reg)
	la,rf=best_model
	param=rf.get_params(deep=True)
	param['max_samples']=param['max_samples']/num_train
	best_model=param,la.get_params()['alpha']
	rfs.append(best_model),	parameters.append(param),val_acc.append(best_score),train_acc.append(best_RERFs_train_acc)
	
	
	out_n_estimators.append(param['n_estimators']),out_max_features.append(param['max_features']),out_min_samples_split.append(param['min_samples_split'])
	out_min_samples_leaf.append(param['min_samples_leaf']),out_max_samples.append(param['max_samples']),out_max_depth.append(param['max_depth'])
	out_min_impurity_decrease.append(param['min_impurity_decrease']),out_ccp_alpha.append(param['ccp_alpha']),out_lambda.append(best_model[1])

	y_pred = la.predict(X_val)+rf.predict(X_val)
	u,v=((y_val-y_pred)**2).sum(),((y_val-y_val.mean())**2).sum()
	out_val_RERFs.append(1-u/v)
	y_pred = la.predict(X_train)+rf.predict(X_train)
	print('RERFs acc : ' ,1-u/v,' with reg : ',best_model[1],'and : ',param,'\n\n')
	u,v=((y_train-y_pred)**2).sum(),((y_train-y_train.mean())**2).sum()
	out_train_RERFs.append(1-u/v)
	#print('RERFs : forest parameters  : \n\t\t\t\t',param,'\n\n')

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

idxi,idxir,idxil = np.argmin((out_val_RERFs-np.median(out_val_RERFs))**2),np.argmin((out_ridge_val-np.median(out_ridge_val))**2),np.argmin((out_val_lasso-np.median(out_val_lasso))**2)

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


print('n_estimator=',out_n_estimators[idxi],t,', max_depth = ',out_max_depth[idxi],', min_samples_split = ',out_min_samples_split[idxi],', min_samples_leaf = ',out_min_samples_leaf[idxi],', max_samples = ',out_max_samples[idxi],', min_impurity_decrease = ',out_min_impurity_decrease[idxi],', out_ccp_alpha : ',out_ccp_alpha[idxi])
print()


print('train_acc ridige optimizei',np.median(out_ridge_train))
print('train acc RERFs',out_train_RERFs)


print('val_acc ridge optimize ' ,np.median(out_ridge_val))
print('val acc RERFs', np.median(out_val_RERFs))
print('val acc RERFs',out_val_RERFs)
print('out_ridge_val',out_ridge_val)

rf,la=rfs[idxi]
np.savez('parameters_gsure.npz',reg_lasso=out_lasso_best_reg[idxil],RERFs_param_forest = rf,RERFs_param_lasso=la,reg_ridge = out_alpha_ridge[idxir])

