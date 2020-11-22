# Self Organizing Map

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

np.random.seed(0)

# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
n_samples = 1500

blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
x, y = blobs

class KMeans():
    def __init__(self, n_clusters, data):
        self.n_clusters = n_clusters
        self.data = data
        self.n_points = np.shape(data)[0]
        self.clusters = np.zeros(self.n_points,dtype = np.int8)
        self.centroids = np.zeros((n_clusters,2))
    def assign_clusters_step(self):
        for i in range(self.n_points):
            self.clusters[i] = self.closest_centroid(self.data[i])  
    def predict_clusters(self):
        switch_centroid = True
        # Randomly initialize the centroids
        for i, val in enumerate(np.random.randint(1,self.n_points,self.n_clusters).tolist()):
            self.centroids[i,:] = self.data[val,:]
        while(switch_centroid):
            switch_centroid = False
            for i in range(self.n_points):
                new_centroid = self.closest_centroid(self.data[i])
                if(new_centroid != self.clusters[i]):
                    switch_centroid = True
                    self.clusters[i]=new_centroid
            #print(self.centroids)
            self.calc_centroids()
            #print(self.centroids)
        self.plot_clusters()
        return self.clusters
    
    def closest_centroid(self, point):
        mini = 0
        minv = np.sqrt(np.square(point[0]-self.centroids[0][0])+np.square(point[1]-self.centroids[0][1]))
        for i in range(1,self.n_clusters):
            dist = np.sqrt(np.square(point[0]-self.centroids[i][0])+np.square(point[1]-self.centroids[i][1]))
            if(dist<minv):
                minv = dist
                mini = i
        return mini
    def calc_centroids(self):
        self.centroids = np.zeros((self.n_clusters,2))
        cluster_count = np.zeros(self.n_clusters)
        for p in range(self.n_points):
            self.centroids[self.clusters[p]][0] += self.data[p][0]
            self.centroids[self.clusters[p]][1] += self.data[p][1]
            cluster_count[self.clusters[p]] += 1
        for i,x in np.ndenumerate(self.centroids):
            self.centroids[i] /= cluster_count[i[0]]
    def set_centroids(self, centroid_points):
        self.centroids = centroid_points
    def get_centroids(self):
        return(self.centroids)
    def plot_clusters(self):
        plt.scatter(self.data[:,0], self.data[:,1], c = self.clusters)
        plt.show()





kdata = np.array([[1,4], [1,3], [0,4], [5,1], [6,2], [4,0]])
initialcentroids = np.array([[1,4], [1,3]])
                            


kmeans = KMeans(2,kdata)
kmeans.set_centroids(initialcentroids)
kmeans.assign_clusters_step()
kmeans.plot_clusters()
kmeans.calc_centroids()
newcentroids = kmeans.get_centroids()
kmeans.assign_clusters_step()
kmeans.plot_clusters()



kmeans = KMeans(2,kdata)
kmeans.set_centroids(initialcentroids)
clusters = kmeans.predict_clusters()
newcentroids = kmeans.get_centroids()
kmeans.plot_clusters()



a = np.arange(6).reshape(3,2)
a

for i,x in np.ndenumerate(a):
    print(i[0])
    print("i: ",i," a: ",a[i])
    #print("x: ",x)

plt.scatter(x[:,0], x[:,1], c = newcentroids)
plt.show()

