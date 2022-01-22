import numpy as np 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import sys
from lasso import lasso
###########################################
#import csv 
#data = []
#with open('./yXtest.csv',newline='') as csvfile:
#	spamreader = csv.reader(csvfile,delimiter = ',')
#	for row in spamreader:
#		data.append(row)
#data=np.asmatrix(data)
#data = np.array(data[1:],dtype=float)
#data = data[:,1:]
#y = data[:,0]
#X = data[:,1:]
#np.savez('./test.npz',y=y , X=X)
###########################################

data = np.load('test.npz')

X , y =data['X'],data['y']

data = np.load('model.npz',allow_pickle=True)

lasso = data['lasso'].item()

a,m_lasso ,std_lasso = lasso['alpha'],lasso['m'],lasso['std']

X_lasso=(X-m_lasso)/std_lasso

random_forest = data['rf'].item()



rf,m_forest,std_forest,idx = random_forest['rf'],random_forest['m'],random_forest['std'],random_forest['idx']

print(rf.get_params(deep=True))

X_forest=((X_lasso[:,idx]-m_forest)/std_forest)

y_pred = np.dot(X_lasso,a)+rf.predict(X_forest)

acc = 1-((y-y_pred)**2).sum()/((y-y.mean())**2).sum()

rows,columns = (len(idx)),1

fig,axe_array = plt.subplots(rows,columns,squeeze=False)
for i,ax_row in enumerate(axe_array):
	for j,axes in enumerate(ax_row):
		index = np.argsort(X[:,i])
		axes.semilogx(X[index,i],y[index],'or',label=' true')
		axes.semilogx(X[index,i],y_pred[index],'ob',label=' pred')


plt.legend()
plt.show()

print('accuracy : ' , acc)

print(rf.feature_importances_)

