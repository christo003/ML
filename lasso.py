import numpy as np
import timeit
def grad(a,X,y,l):
    k=0
    G=np.dot(X.T,(np.dot(X,a)-y))
    R=np.zeros(a.shape[0])
    for i in a:
        if i !=0:
            R[k]=(G[k]+l*np.sign(i))
        else:
            if np.abs(G[k])<=l:
                R[k]=0
            if G[k]>l:
                R[k]=(G[k]-l)
            if G[k]<-l:
                R[k]=(G[k]+l)
        k=k+1
    return R

def lasso(X,y,a,l):
    l=l/2 #le l/2 est enlever pour faire le lasso path
    start = timeit.default_timer()
    iter_max=10**4#10**5+5*10**4
    i = 0
    tol1=1/2*np.linalg.norm(y-np.dot(X,a),2)**2+l*np.linalg.norm(a,1)
    tol2=2
    tol=np.abs(tol1-tol2)
    eps=10**(-3)
    if (a!=np.zeros(a.shape[0])).all():
        g=grad(a,X,y,l)
        i=np.argmax(np.abs(g))
    k=0
    while (tol>eps)&(k<iter_max):
        x= X.T[i]
        a_i=np.delete(a,i,axis=0) 
        X_i=np.delete(X,i,axis=1)
        z_hat=y-np.dot(X_i,a_i)
        proj=np.vdot(x,z_hat)
        x_norm=np.vdot(x,x)
        a[i]=np.sign(proj)*np.max((np.abs(proj)-l)/x_norm,0)
        tol2=tol1
        tol1=1/2*np.linalg.norm(y-np.dot(X,a),2)**2+l*np.linalg.norm(a,1)
        tol=np.abs(tol1-tol2)/eps
        g=grad(a,X,y,l)
        i=np.argmax(np.abs(g))
        k=k+1
    stop = timeit.default_timer()
    #print('lambda is : ',l)
    #print('      tol is : ',tol)
    #print('      num iter is : ',k)
    #print('      Time: ', stop - start)  
    return a,tol,k,stop - start
def warm_start(X,y,num_lasso_path):
	X=((X.T-np.mean(X,1))/np.std(X,1)).T
	k=0
	lmax=np.max(np.dot(X.T,y))
	L=np.linspace(lmax,0,num_lasso_path)
	A=np.zeros((num_lasso_path+1,X.shape[1]))
	size_l=L.shape[0]
	L[0]=lmax-0.1
	TOL=np.zeros(num_lasso_path+1)
	NUM_ITER=np.zeros(num_lasso_path+1)
	TIME=np.zeros(num_lasso_path+1)
	for l in L:
	    A[k+1],TOL[k+1],NUM_ITER[k+1],TIME[k+1]=lasso(X,y,A[k],l)
	    k=k+1
	return A,L,TOL,NUM_ITER,TIME
