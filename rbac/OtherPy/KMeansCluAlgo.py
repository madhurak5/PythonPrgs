import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


df = pd.DataFrame(
    {
        'x' : [12, 20, 28, 18, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72],
        'y' : [39, 36, 30, 52, 54, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19,  7, 24]
    }
)
print(df)
# rndSeed = np.random.seed(200)
# print(rndSeed)
k = 2
# centroids = {
#     i+1 : [np.random.randint(0, 80), np.random.randint(0, 80)]
#     for i in range(k)
# }
# fig = plt.figure(figsize=(5, 5))
# plt.scatter(df['x'], df['y'], color = 'k')
# colmap = {1:'r', 2:'g', 3:'b'}
# for i in centroids.keys():
#     plt.scatter(*centroids[i], color = colmap[i])
# plt.xlim(0, 80)
# plt.ylim(0, 80)
# plt.show()
#
kmeans = KMeans(n_clusters=3)
print ("Kmeans after fitting : \n", kmeans.fit(df))
labels = kmeans.predict(df)
print("Labels : \n", labels)
centroids = kmeans.cluster_centers_
print("Centroids : ", centroids)
fig = plt.figure(figsize=(5, 5))
plt.scatter(df['x'], df['y'], color = 'k')
colmap = {1:'r', 2:'g', 3:'b'}


plt.xlim(0, 80)
plt.ylim(0, 80)
plt.show()