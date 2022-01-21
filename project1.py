import numpy as np 
import sys 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from lasso import lasso

data=np.load('regression_data1.npz')
y,X= data['y'],data['X']
num_data,num_feature=X.shape


num_lasso =5
num_cv=10

num_fold=int(num_data/num_cv)


I_all=[np.arange(num_data) for i in range(num_cv)]


for i in range(num_cv):
	np.random.shuffle(I_all[i])



out_n_estimators,out_max_features,out_max_depth,out_max_depth,out_min_samples_split,out_min_samples_leaf,out_bootstrap, out_ccp_alphas = [],[],[],[],[],[],[],[]

n_estimators=[140,150]
max_features = ['log2','sqrt']
max_depth=[2,3,4]
#max_depth.append(None)
min_samples_split = [4,5]
min_samples_leaf = [2,3]
bootstrap = [True]
ccp_alphas = [0]
random_grid = {'n_estimators':n_estimators,'max_features':max_features,'max_depth':max_depth,'min_samples_split':min_samples_split,'min_samples_leaf' : min_samples_leaf,'bootstrap':bootstrap}
defaut_train,default_val=[],[]
out_lambda=[]
best=[]
rfs,val_acc,train_acc = [],[],[] 
out_ridge_train,out_ridge_val,out_alpha_ridge = [],[],[]
out_train_RERFs,out_val_RERFs=[],[]
REG=np.linspace(1662.0,1663,num_lasso)

for j in range(num_cv):
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
	X_train = (X_train - m_train ) / std_train
	
	X_val = (X_val - m_train) / std_train
	
	best_ridge=-sys.maxsize
	for ridge in [1,100,1000]:#,100,1000]:
		rr = Ridge(alpha=ridge)
		rr.fit(X_train,y_train)
		score_ridge = rr.score(X_val,y_val)
		if score_ridge>best_ridge:
			best_ridge = score_ridge
			train_ridge = rr.score(X_train,y_train)
			alpha_ridge = ridge
	out_ridge_train.append(train_ridge)
	out_ridge_val.append(best_ridge)
	out_alpha_ridge.append(alpha_ridge)

	
	alpha = np.zeros(num_feature)
	best_lasso_score=-sys.maxsize
	
	for k in  range(num_lasso) :
		print('\t num_lasso : ',k+1, ' / ', num_lasso)
		#reg = np.exp(np.log(1/(10*num_lasso))+k*(np.log(num_lasso)-np.log(1/(10*num_lasso)))/(num_lasso-1))
		reg = REG[k]
		alpha,_,_,_ = lasso(X_train,y_train,np.zeros(num_feature),reg)
		y_pred_val = np.dot(X_val,alpha)
		y_pred_train = np.dot(X_train,alpha)

		u = ((y_val-y_pred_val)**2).sum()
		v = ((y_val-y_val.mean())**2).sum()
		lasso_score=1-u/v
		if best_lasso_score<lasso_score:
			best_lasso_score=lasso_score
			best_lasso_reg=reg
	
		idx_no_zero = np.arange(num_feature)[alpha!=0]
		residual = (y_train- np.dot(X_train,alpha))
		val_target_forest =( y_val-np.dot(X_val,alpha))
		print('idx_no_zero',idx_no_zero)

		
		train_target_forest = residual
		
		train_forest = X_train[:,idx_no_zero]
		val_forest = X_val[:,idx_no_zero]
		alpha_forest=alpha[idx_no_zero]

		num_idx_no_zero=len(idx_no_zero)
		m_forest,std_forest  = np.mean(train_forest,0).reshape((1,num_idx_no_zero)),np.std(train_forest,0).reshape((1,num_idx_no_zero))
		train_forest=(train_forest-m_forest)/std_forest
		val_forest=(val_forest-m_forest)/std_forest

		best_forest_score=-sys.maxsize
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
							for boot in bootstrap:
								#print('\t\t      bootstrap ',boot, ' / ',bootstrap )
								for ccp_alpha in ccp_alphas:
									#print('\t\t       ccp_alpha ',ccp_alpha, ' / ',ccp_alphas)
									rf = RandomForestRegressor(n_estimators=n_estimator,max_features=max_feature,max_depth=depth,min_samples_split=min_sample_split,min_samples_leaf=min_sample_leaf,bootstrap=boot)
									rf.fit(train_forest,train_target_forest)
									score=rf.score(val_forest,val_target_forest)
									if score>best_forest_score:
										best_forest_score=score
										best_rf = rf	
										best_train_forest_acc=rf.score(train_forest,train_target_forest)

									y_pred = y_pred_val+(rf.predict(val_forest))
									u = ((val_target_forest-y_pred)**2).sum()
									v = ((val_target_forest-val_target_forest.mean())**2).sum()
									score = 1-u/v

									if score>best_score:
										best_score=score
										best_RERFs={'lasso_reg':reg,'tree_reg':rf.get_params(deep=True)}
										y_pred = y_pred_train+rf.predict(train_forest)
										u = ((train_target_forest-y_pred)**2).sum()
										v = ((train_target_forest-train_target_forest.mean())**2).sum()
										best_RERFs_train_acc = 1-u/v
										best_model = [alpha_forest,rf,idx_no_zero,m_train,std_train,m_forest,std_forest,reg]

		rf = RandomForestRegressor()
		rf.fit(train_forest,train_target_forest)
		default_val.append(rf.score(val_forest,val_target_forest))
		defaut_train.append(rf.score(train_forest,train_target_forest))
	alpha,rf,idx_no_zero,m_train,std_train,m_forest,std_forest,reg=best_model
	rfs.append(best_model)
	val_acc.append(best_score)
	train_acc.append(best_RERFs_train_acc)
	param = rf.get_params(deep=True)
	out_n_estimators.append(param['n_estimators'])
	out_max_features.append(param['max_features'])
	tot_depth=0
	for rt in rf.estimators_:
		tree = rt.tree_
		n_nodes = tree.node_count
		children_left = tree.children_left
		children_right = tree.children_right
		node_depth = np.zeros(n_nodes,dtype=np.int64)
		is_leaves = np.zeros(n_nodes,dtype=bool)
		stack = [(0,0)]
		while len(stack)>0:
			node_id,depth = stack.pop()
			node_depth[node_id]=depth	
			is_split_node = children_left[node_id]!=children_right[node_id]
			if is_split_node:
				stack.append((children_left[node_id],depth+1))
				stack.append((children_right[node_id],depth+1))
			else:
				is_leaves[node_id] = True
		tot_depth=tot_depth+depth
	tot_depth /=len(rf.estimators_)
	out_max_depth.append(tot_depth)
	out_min_samples_split.append(param['min_samples_split'])
	out_min_samples_leaf.append(param['min_samples_leaf'])
	out_bootstrap.append(param['bootstrap'])
	out_ccp_alphas.append(param['ccp_alpha'])
	best.append(rf)
	out_lambda.append(reg)
	y_pred = np.dot(X_val[:,idx_no_zero],alpha)+rf.predict((X_val[:,idx_no_zero]-m_forest)/std_forest)
	u=((y_val-y_pred)**2).sum()
	v = ((y_val-y_val.mean())**2).sum()
	out_val_RERFs.append(1-u/v)
	y_pred = np.dot(X_train[:,idx_no_zero],alpha)+rf.predict((X_train[:,idx_no_zero]-m_forest)/std_forest)
	u=((y_train-y_pred)**2).sum()
	v = ((y_train-y_train.mean())**2).sum()
	out_train_RERFs.append(1-u/v)
	


print('out_alpha_ridge',out_alpha_ridge)
print('out_ridge_train',out_ridge_train)

print('out_lambda',out_lambda)
#print('defaut train accu' ,defaut_train)
#print('defaut val accu'  ,default_val)
#print('train_acc',train_acc)
#print('val_acc',val_acc)
print()
print('out_n_estimators',out_n_estimators)
print('out_max_features',out_max_features)
print('out_max_depth',out_max_depth)
print('out_min_samples_split',out_min_samples_split)
print('out_min_samples_leaf',out_min_samples_leaf)
#print('out_bootstrap',out_bootstrap)
#print('out_ccp_alpha',out_ccp_alphas)

print()
print('reg ridge',np.median(out_alpha_ridge))
print()
print('lambda',np.median(out_lambda))
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
print('n_estimator=',np.median(out_n_estimators),t,', max_depth = ',np.median(out_max_depth),', min_samples_split = ',np.median(out_min_samples_split),', min_samples_leaf = ',np.median(out_min_samples_leaf))
print()



print('train accu tree default' ,np.median(defaut_train))
print('train_acc tree optimize',np.median(train_acc))
print('train_acc ridige optimizei',np.median(out_ridge_train))
print('train acc RERFs',out_train_RERFs)

#print('defaut val accu tree default'  ,np.median(default_val))
print('val_acc tree optimize',np.median(val_acc))
print('val_acc ridge optimize ' ,np.median(out_ridge_val))
print('val acc RERFs', np.median(out_val_RERFs))
print('val acc RERFs',out_val_RERFs)
print('out_ridge_val',out_ridge_val)
