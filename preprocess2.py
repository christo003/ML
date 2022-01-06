import numpy as np
from lasso import lasso
data = np.load('regression_data2.npz')
X,y = data['X'],data['y']
print(np.mean(X))
print(np.mean(y))
