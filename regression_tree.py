import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import sys
num_train=1000
num_test =100 
reg,num_leaf=100,20
def f(I):
	out=[]
	for x in I:
		if x <0.2:
			y=5
		if 0.2<=x<0.4:
			y=10
		if 0.4<=x<0.7:
			y=1
		if 0.7<= x:
			y=8
		out.append(y)
	return np.array(out)
def g(I):
	out = []
	for x in I:
		y=0
		if (x[1]<0.2):
			if(x[0]<0.1):
				y=10
			else : 
				y=-6#y=(x[1]+2)**2
		if (0.2<=x[0]<0.4):
			if(0.1<=x[1]):
				y=5
				if x[1]<=0.5: 
					y = 3#np.sin(x[0]+x[1])
				else : 
					y=-4
		if (0.4<=x[1]<0.8):
			if(0.5<=x[0]<0.7):
				y=6
			else:
				y = 2#np.exp(x[0]+1)
		if (0.8<=x[0]):
			if(0.7<=x[1]):
				y=5
			if x[1]<0.4 : 
				y=2
			else : 	
				y=-5
		out.append(y)

	return np.array(out)

g = lambda x : np.sin(4*(x[:,0]+x[:,1]))

def rm_key(dictionary,key):
	if not isinstance(key,list):
		key = [key]
	_dict = dictionary.copy()
	for k in key:
		_dict.pop(k,None)
	return _dict

train = np.random.uniform(0,1,(num_train,2))
target = g(train)+np.random.normal(0,0.2,num_train)
fig = plt.figure(figsize=plt.figaspect(0.5))
ax=plt.axes(projection='3d')
ax.plot3D(train[:,0],train[:,1],target,'oc',label='train')

def regression_tree_bis(train,target,num_leaf):
	if len(train.shape)>1:
		num_data,num_feature = train.shape
	else:
		num_data,num_feature=train.shape[0],1
	res,idx,threshold= sys.maxsize,0,0
	idx_sort = [np.argsort(train[:,k]) for k in range(num_feature)]
	train_sort,target_sort = [train[idx] for idx in idx_sort],[target[idx] for idx in idx_sort]
	I=np.arange(num_feature)
	np.random.shuffle(I)
	#for k in  I :
	for i in range(np.random.randint(0,int(num_data/3),1)[0],np.random.randint(num_data-int(num_data/3),num_data,1)[0],num_leaf):
		#idx=np.argsort(train[:,k])
		#x,y = train[idx],target[idx]
		#for i in range(1,num_data-1,num_leaf):
		for k in I:
			x,y = train_sort[k],target_sort[k]
			mean1,mean2 = np.mean(y[:i]),np.mean(y[i:])
			t1,t2 = np.sum((y[:i]-mean1)**2),np.sum((y[i:]-mean2)**2)
			current=t1+t2
			if current<res:
				threshold,x1,x2,y1,y2,m1,m2,res1,res2,feature=(x[i,k]+x[i+1,k])/2,x[:i],x[i:],y[:i],y[i:],mean1,mean2,t1,t2,k
	return threshold,x1,x2,y1,y2,m1,m2,res1,res2,feature 
def regression_tree(x,y,num_leaf,reg=0,max_deep = 200,subset_var=False,dist_to_root=0,res_prev=sys.maxsize):
	if not subset_var :
		var = np.arange(x.shape[1])
	else:
		var=subset_var[dist_to_root]
		max_deep = len(subset_var)
	node = {'is_leaf':False,'dist_to_root':dist_to_root}#,'num_child':0}
	if (len(x)>num_leaf)&(dist_to_root<max_deep):
		threshold,x1,x2,y1,y2,m1,m2,res1,res2,feature = regression_tree_bis(x[:,var],y,num_leaf)
		node['threshold'],node['feature']=threshold,feature
		direction = np.arange(2)
		np.random.shuffle(direction)
		for i in direction:
			if i == 0:
				left_child= regression_tree(x1,y1,num_leaf,reg,max_deep,subset_var,dist_to_root+1,res1)
				
				res_reg1 = res1+reg
				left_child['m'] = m1
				if (left_child['is_leaf']):
					if(res_prev>res_reg1):
						left_child['res'] = res_reg1
					else:
						left_child['res']=res_prev+reg
						
				node['left'] = left_child
				
			else:
				res_reg2 = res2+reg
				right_child = regression_tree(x2,y2,num_leaf,reg,max_deep,subset_var,dist_to_root+1,res2)
				right_child['m'] = m2
				if (right_child['is_leaf']):
					if(res_prev>res_reg2):
						right_child['res'] = res_reg2
					else:
						right_child['res']=res_prev+reg
				node['right'] = right_child

	else : 
		node['is_leaf'] = True

	return node

	
	 


def predict(x):
	node=tree
	y = np.zeros(x.shape[0])
	for i in range(x.shape[0]):
		while(not node['is_leaf']):
			if x[i,node['feature']] <node['threshold']:
				node = node['left']
			else :
				node = node['right']
		y[i] = node['m']
		node=tree
	return y
def max_deep(tree,n):
	out=[]
	for node in [tree['left'],tree['right']]:
		if not node['is_leaf']:
			n=max_deep(node,n)
		else :
			n=node['dist_to_root']
		out.append(n)
	return max(out)
def display(tree):
	n = max_deep(tree,0)**2
	list_node = [tree]
	space=''
	 
	while(len(list_node)>0):
		space=''
		for i in range(n):
			space+=' '	
		out=[]
		p=' '
		for node in list_node:
			#p+=space
			for cle ,valeur in rm_key(node,['left','right','m','dist_to_root','is_leaf']).items():	
				if str(cle)=='feature':	
					p=p+str(cle) +str(valeur)+'  ' +space
				else:
					p=p+str(cle)+' '+str(round(valeur,3))+' '
			
			if not node['is_leaf']:
				out.append(node['left'])
				out.append(node['right'])			
		print(p+'\n')
		n-=n
		list_node=out
tree= regression_tree(train,target,num_leaf,reg=reg,max_deep=20)
#print(tree)
test_train = np.random.uniform(0,1,(num_test,2))
test_target = g(test_train)
predict = predict(test_train)
#print(predict)
ax.plot3D(test_train[:,0],test_train[:,1],test_target,'or',label='test')
ax.plot3D(test_train[:,0],test_train[:,1],predict,'ob',label='predict')
plt.legend()
plt.show()
#print(max_deep(tree,0))
display(tree)
