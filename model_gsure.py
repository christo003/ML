import numpy as np
import sklearn as sk 
import sys 

test = np.load('test.npz')
X_test,y_test= test
train1 = np.load('regression_data1.npz')
X_train1,y_train2 = train1
train2 = np.load('regression_data2.npz')
X_train2,y_train2 = train2
lasso = np.load('model_lasso.npz')
alpha,reg,X_lasso,y_lasso,mean,std=lasso

lasso= sk.linear_model.Lasso(alpha=reg)

