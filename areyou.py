
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering


n_samples = 1500
random_state = 170
x,y = make_blobs(n_samples=n_samples,cluster_std=[1.0,3.5,0.5,0.4,1.2],random_state=random_state,centers=5)
#x,y = make_blobs(n_samples=n_samples,random_state=random_state,centers=5)


#transformation = [[0.5,0.5],[-0.4,-0.8]]
#x = np.dot(x,transformation)

y_pred = KMeans(random_state=random_state).fit_predict(x)


#ans = MeanShift(bandwidth=2).fit(x)



plt.figure()
plt.subplot(151)
plt.scatter(x[:,0],x[:,1])
plt.title("data")
plt.subplot(152)
plt.title("Kmean")
plt.scatter(x[:,0],x[:,1],c=y_pred)
plt.subplot(153)
plt.title("Meanshift")
y_pred = MeanShift(bandwidth=2).fit_predict(x)
plt.scatter(x[:,0],x[:,1],c=y_pred)
plt.subplot(154)
plt.title("DBSCAN")
y_pred = DBSCAN(eps=1, min_samples=2).fit_predict(x)
plt.scatter(x[:,0],x[:,1],c=y_pred)
plt.subplot(155)
plt.title("AgglomerativeClustering")
y_pred = AgglomerativeClustering().fit_predict(x)
plt.scatter(x[:,0],x[:,1],c=y_pred)
#plt.legend()
plt.show()