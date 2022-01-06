import numpy as np
import csv
import matplotlib.pyplot as plt 

data=[]
with open('./regression_data.csv',newline='') as csvfile:
	spamreader= csv.reader(csvfile,delimiter = ',')
	for row in spamreader: 
		data.append(row)
data = np.asmatrix(data)
data =np.array(data[1:],dtype=float)
data = data[:,1:]
y = data[:,0]
X = data[:,1:]
np.savez('./regression_data1.npz',y=y,X=X)


########### visualisation  #################

xedges,yedges=np.linspace(0,1,11),np.linspace(0,1,11)
from matplotlib.image import NonUniformImage
y_scale = (y+np.abs(np.min(y)))/(np.max(y)+np.abs(min(y)))
sort_idx = np.argsort(y)
y_sort = y_scale[sort_idx]
X_arrange=((X+np.abs(np.min(X,1).reshape((X.shape[0],1))))/(np.max(X,1)+np.abs(np.min(X,1))).reshape((X.shape[0],1)))[sort_idx]
columns = 10
rows = 5
fig, ax_array = plt.subplots(rows, columns,squeeze=False)
k=0
for i,ax_row in enumerate(ax_array):
	for j,axes in enumerate(ax_row):
		axes.set_title(str(k).format(i,j))
		axes.set_yticklabels([])
		axes.set_xticklabels([])
		H, xedges,yedges=np.histogram2d(X_arrange[:,k], y_sort ,bins=(xedges,yedges))
		H = H.T
		#axes.imshow(H, interpolation='nearest', origin='lower',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
		im = NonUniformImage(axes, interpolation='bilinear')
		xcenters = (xedges[:-1] + xedges[1:]) / 2
		ycenters = (yedges[:-1] + yedges[1:]) / 2
		im.set_data(xcenters, ycenters, H)
		axes.images.append(im)
		k=k+1
plt.savefig('visu_feature.png')
plt.show()

