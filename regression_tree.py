import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import sys
num_train=50
num_test = 50
reg,num_leaf=0,5
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
		if (x[0]<0.2):
			if(x[1]<0.1):
				y=10
			else : 
				y=-6#y=(x[1]+2)**2
		if (0.2<=x[0]<0.4):
			if(0.1<=x[1]):
				y=5
				if x[1]<=0.5: 
					y = 10#np.sin(x[0]+x[1])
				else : 
					y=-4
		if (0.4<=x[0]<0.8):
			if(0.5<=x[1]<0.7):
				y=10
			else:
				y = 2#np.exp(x[0]+1)
		if (0.8<=x[0]):
			if(0.7<=x[1]):
				y=5
			if x[1]<0.4 : 
				y=2
			else : 	
				y=20
		out.append(y)
	return np.array(out)


train = np.random.uniform(0,1,(num_train,2))
target = g(train)+np.random.normal(0,1,num_train)
fig = plt.figure(figsize=plt.figaspect(0.5))
#ax=fig.add_subplot(1,2,1,projection='3d')
ax=plt.axes(projection='3d')
ax.plot3D(train[:,0],train[:,1],target,'oc',label='train')
def regression_tree_bis(x,y,num_leaf):
	if len(x.shape)>1:
		num_data,num_feature = x.shape
	else:
		num_data,num_feature=x.shape[0],1
	res,idx,threshold= sys.maxsize,0,0
	I=np.arange(num_feature)
	np.random.shuffle(I)
	for k in I:
		idx_sort = np.argsort(x[:,k])
		x,y = x[idx_sort],y[idx_sort]
		
		for i in range(1,num_data-1,num_leaf):
			mean1,mean2 = np.mean(y[:i]),np.mean(y[i:])
			t1,t2 = np.sum((y[:i]-mean1)**2),np.sum((y[i:]-mean2)**2)
			current=t1+t2
			if current<res:
				threshold,x1,x2,y1,y2,m1,m2,res1,res2,feature=(x[i,k]+x[i+1,k])/2,x[:i],x[i:],y[:i],y[i:],mean1,mean2,t1,t2,k
	return threshold,x1,x2,y1,y2,m1,m2,res1,res2,feature 
def regression_tree(x,y,num_leaf,reg=0,dist_to_root=0,res_prev=sys.maxsize):
	node = {'is_leaf':False,'dist_to_root':dist_to_root}#,'num_child':0}
	if len(x)>num_leaf:
		threshold,x1,x2,y1,y2,m1,m2,res1,res2,feature = regression_tree_bis(x,y,num_leaf)
		node['threshold'],node['feature']=threshold,feature
		direction = np.arange(2)
		np.random.shuffle(direction)
		dist_to_root+=1
		for i in direction:
			if i == 0:
				left_child= regression_tree(x1,y1,num_leaf,reg,dist_to_root,res1)
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
				right_child = regression_tree(x2,y2,num_leaf,reg,dist_to_root,res2)
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

	
	 

tree= regression_tree(train,target,num_leaf,reg)
print(tree)
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
def display_bis(layer,n):
	out=[]
	s = ' '
	c = ''
	for i in range(n):
		c+=s
	p=' '
	if n >0:
		for node in layer:
			if not node['is_leaf']:
				out.append(node['left'])
				out.append(node['right'])
				p=p+'\t'+str(node['threshold'])+' ' +str(node['feature'])+' ' +str(node['m'])
			else : 
				p=p+' '+str(node['m'])
		n-=1
		print(c+p)
		display_bis(out,n)
	return out,n
def display(tree):
	m = max_deep(tree,0)
	layer = [tree['left'],tree['right']]
	display_bis(layer,m)
test_train = np.random.uniform(0,1,(num_test,2))
test_target = g(test_train)
predict = predict(test_train)
#print(predict)
#ax=fig.add_subplot(1,2,2,projection='3d')
ax.plot3D(test_train[:,0],test_train[:,1],test_target,'or',label='test')
ax.plot3D(test_train[:,0],test_train[:,1],predict,'ob',label='predict')
plt.legend()
plt.show()
print(max_deep(tree,0))
display(tree)
