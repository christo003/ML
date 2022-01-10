import numpy as np
import matplotlib.pyplot as plt
import sys
n=1000
num_leaf=20
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
feature1 = np.sort(np.random.uniform(0,1,n))

feature2=feature1[200:400]
feature2=np.concatenate((feature2,feature1[700:]))
feature2=np.concatenate((feature2,feature1[0:200]))
feature2=np.concatenate((feature2,feature1[400:700]))
target = f(feature1)+np.random.normal(0,0.2,n)
plt.figure()
plt.plot(feature1,target,'ro')
plt.plot(feature2,target,'bo')
plt.show()
def regression_tree_bis2(train,target,num_leaf,r,s,rec):
	idx_min = np.argmin(r)
	res= r[idx_min]
	threshold,x1,x2,y1,y2,m1,m2=rec[idx_min]
	u=0
	for k in s:
		idx_sort,n = np.argsort(train[:,k]),train.shape[0]
		x,y = train[idx_sort,:],target[idx_sort]

		for i in range(1,n-1,num_leaf):
			mean1,mean2 = np.mean(y[:i]),np.mean(y[i:])
			current = np.sum((y[:i]-mean1)**2)+np.sum((y[i:]-mean2)**2)
			if current<res:
				threshold,x1,x2,y1,y2,m1,m2=(x[i,k]+x[i+1,k])/2,x[:i],x[i:],y[:i],y[i:],mean1,mean2
				rec[k]=[threshold,x1,x2,y1,y2,m1,m2]
				r[k]=res
				res,idx_min,umin =current,k,u
		u+=1
	del(rec[idx_min])
	del(r[idx_min])
	return threshold,x1,x2,y1,y2,m1,m2,r,[idx_min],rec
def regression_tree2(train,target,num_leaf,start=True):
	r,s,rec= [],[],[]
	#r,s,rec=np.zeros(train.shape[1]).tolist(),np.arange(train.shape[1]).tolist(),[[0,0,0,0,0,0,0] for i in range(train.shape[1])]
	print(train)
	if isinstance(train,np.ndarray):
		if len(train.shape)>1:
			for k in range(train.shape[1]):
				threshold,x1,x2,y1,y2,m1,m2,res = regression_tree_bis(train[:,k],target,num_leaf)
				r.append(res),s.append(k),rec.append([threshold,x1,x2,y1,y2,m1,m2])
	
		u=0
		for k in s:
			if len(train[:,k])<num_leaf:
					s=np.delete(s,u)
			u+=1
		if len(s)>0:
			left,right = [],[]
			threshold,x1,x2,y1,y2,m1,m2,r,s,rec = regression_tree_bis2(train,target,num_leaf,r,s,rec)
			left.append(threshold),left.append(m1),left.append(regression_tree2(x1,y1,num_leaf,False))
			right.append(threshold),right.append(m2),right.append(regression_tree2(x2,y2,num_leaf,False))
			return left,right

def regression_tree_bis(x,y,num_leaf):
	idx_sort,n = np.argsort(x),x.shape[0]
	x,y = x[idx_sort],y[idx_sort]
	res,idx,threshold= sys.maxsize,0,0
	for i in range(1,n-1,num_leaf):
		mean1,mean2 = np.mean(y[:i]),np.mean(y[i:])
		current = np.sum((y[:i]-mean1)**2)+np.sum((y[i:]-mean2)**2)
		if current<res:
			threshold,x1,x2,y1,y2,m1,m2,res=(x[i]+x[i+1])/2,x[:i],x[i:],y[:i],y[i:],mean1,mean2,current
	return threshold,x1,x2,y1,y2,m1,m2,res	 
def regression_tree(x,y,num_leaf,start=True):
	left,right = {'threshold':np.array([]),'m':np.array([])},{'threshold':np.array([]),'m':np.array([])}

	if len(x)>num_leaf:
		threshold,x1,x2,y1,y2,m1,m2,res = regression_tree_bis(x,y,num_leaf)
		left['threshold']=np.concatenate((left['threshold'],[threshold]))
		left['m']=np.concatenate((left['m'],[m1]))
		l,r = regression_tree(x1,y1,num_leaf,False)
		left['threshold']= np.concatenate((r['threshold'],left['threshold']))
		left['threshold']=np.concatenate((l['threshold'],left['threshold']))
		left['m']=np.concatenate((r['m'],left['m']))
		left['m']=np.concatenate((l['m'],left['m']))

		l,r = regression_tree(x2,y2,num_leaf,False)
		right['threshold']=np.concatenate((l['threshold'],right['threshold']))		
		right['threshold']=np.concatenate((right['threshold'],[threshold]))
		right['threshold']=np.concatenate((right['threshold'],r['threshold']))
		right['m']=np.concatenate((l['m'],right['m']))		
		right['m']=np.concatenate((right['m'],[m2]))
		right['m']=np.concatenate((right['m'],r['m']))
	else : 
		return left,right
	if start :
		tl = np.concatenate((left['threshold'],right['threshold']))
		tr =  np.concatenate((left['m'],right['m']))
		tl,idx=np.unique(tl,return_index=True)
		idx = np.sort(idx)
		left, right= tl,tr[idx]
		#left=np.concatenate(([0],left))
		left=np.concatenate((left,[np.max(x)]))
	else:
		return left,right
	return left,right
	 
	#threshold,x1,x2,y1,y2,m1,m2,res = regression_tree_bis(feature,target,3)
train=np.concatenate((feature1.reshape((n,1)),feature2.reshape((n,1))),1)

t,m = regression_tree2(train,target,num_leaf)

def evaluate(thres,m,x):
	if len(x)>1:
		out=[]
		for xi in x : 
			for i in range(m.shape[0]):
				if thres[i]<=xi<thres[i+1]:
					out.append( m[i])	
	else:
		out=0
		for i in range(m.shape[0]):
			if thres[i]<=x<thres[i+1]:
				out= m[i]	
	return out
print(t)
print(m)
#for i in range(m.shape[0]):
#	plt.plot([t[i],t[i+1]],[m[i],m[i]],'b')
#plt.show()
