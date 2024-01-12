import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import umap
import logging
from sklearn.metrics import  silhouette_score, silhouette_samples
from sklearn.cluster import DBSCAN, KMeans
from sklearn.model_selection import GridSearchCV
import collections
# from ttyplotlib import show

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

class data_class():
    def __init__(self, dir):
        self.dir = dir
        self.test_file = os.path.join(dir, "testing.csv")
        self.train_file = os.path.join(dir, "training.csv")
        self.train_data = pd.read_csv(self.train_file)
        self.test_data = pd.read_csv(self.test_file)
        self.ideal_min_values = []

    def reduce_dimensions_umap(self):
        
        # recommended parameters for clustering 
        fit = umap.UMAP(
            n_neighbors=10, # number of nearest neighbors considered (affects balance between global/local structure, higher value result in more global view)
            min_dist=0.0, # min distance between points in the low-dimensional representation
            n_components=2 # number of dinensions in low-dimensional space
        )

        self.umap = fit.fit_transform(self.train_data)

        return(self.umap)
    
    def find_best_num_clusters(self, min_n_clusters, max_n_clusters):
        # find number of clusters:
        range_n_clusters = np.arange(min_n_clusters, max_n_clusters, 1)
        # dict where we will store keys: numb clusters, value: silhouette score
        numb_clust_silhouette = dict()
        for n_clusters in range_n_clusters:
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            cluster_labels = clusterer.fit_predict(u)

            # silhouette only can be computed when numb of clusters > 1 
            if len(set(cluster_labels)) > 1:
                silhouette_avg = silhouette_score(u, cluster_labels)
            
                numb_clust_silhouette[n_clusters] = silhouette_avg
                


                # get ideal number of clusters based on kmeans and silhouette
        self.best_n_clust = max(numb_clust_silhouette, key=numb_clust_silhouette.get)
        
        logger.info(f"The optimal number of clusters identified using k-means and silhouette is:\n{self.best_n_clust}")
            


        return(self.best_n_clust)


    def get_ideal_eps(self, min_eps, max_eps, step):
        range_eps = np.arange(min_eps, max_eps, step)
        range_min_pts = np.arange(1, 15, 1)
        eps_silhouette = dict()
        for eps in range_eps:
            for min_pts in range_min_pts:
                
                db = DBSCAN(eps = eps, min_samples = min_pts).fit(self.umap)
                labels = db.labels_
                core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
                core_samples_mask[db.core_sample_indices_] = True
                # silhouette score can't be run if only appears one cluster
                if len(set(labels)) > 1:
                    silhouette_avg = silhouette_score(u, labels)
                eps_min_pts = f"{eps}-{min_pts}"
                eps_silhouette[eps_min_pts] = silhouette_avg
                # Create a DataFrame with UMAP coordinates and DBSCAN cluster labels
                # df = pd.DataFrame(data=u, columns=['UMAP_1', 'UMAP_2'])
                # df['DBSCAN_Cluster'] = labels
        
        best_eps_min_pts = max(eps_silhouette, key=eps_silhouette.get)
        print(eps_silhouette)
        self.best_eps = float(best_eps_min_pts.split("-")[0])
        self.ideal_min_values = int(best_eps_min_pts.split("-")[1])
        logger.info(f"Optimal epsilon value is = {self.best_eps} with min points = {self.ideal_min_values}, the silhouette score is:\n {silhouette_avg}")

        self.run_dbscan_over_umap(min_samples=self.ideal_min_values, eps=self.best_eps)
        return(self.best_eps)
    
    def get_ideal_min_samples(self, min_min_samples, max_min_samples, step):
        range_min_samples = np.arange(min_min_samples, max_min_samples, step)

        if not hasattr(self, "best_eps"):
            self.get_ideal_eps()
            
        for min_samples in range_min_samples:
            db = DBSCAN(eps=self.best_eps, min_samples= min_samples).fit(self.umap)

            core_samples_mask = np.zeros_like(db.labels_, dtype = bool)
            core_samples_mask[db.core_sample_indices_] = True

            # ignoring label -1 as it is for outliers
            labels = set([label for label in db.labels_ if label >= 0])

            
            print(len(labels), self.best_n_clust )
            if len(labels) == self.best_n_clust:
                self.ideal_min_values.append(min_samples)
            
            # if there is no match between expected numb of clusters and numb of clusters found by dbscan
            if not self.ideal_min_values:
                min_difference = float("inf")
                closest_nums = []
                # take the closest number of clusters:
                for num in labels:
                    difference = abs(num - self.best_n_clust)
                    if difference < min_difference:
                        min_difference = difference
                        closest_nums = [num]
                    elif difference == min_difference:
                        closest_nums.append(num)
                
                self.ideal_min_values = closest_nums
        
        logger.info(f"Ideals min values obtaining ideal number of clusters: {self.best_n_clust} using ideal epsilon: {self.best_eps} are the following: {min_samples}")
        return(self.ideal_min_values)
    def run_dbscan_over_umap(self, min_samples, eps):
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(self.umap)
        
        
        

        df = pd.DataFrame(data=self.umap, columns=['UMAP_1', 'UMAP_2'])
        df['DBSCAN_Cluster'] = db.labels_

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='UMAP_1', y='UMAP_2', hue='DBSCAN_Cluster', data=df, palette='viridis', legend='full')
        plt.title('UMAP Clustering with DBSCAN')
        plt.show()

    




def get_dirs():
    # obtain current path
    current_path = os.getcwd()

    # get list of all items in the current path
    items_path = os.listdir(current_path)
    logger.info(f"Paths to be explored: {len(items_path)}")


    # filter the directories
    dirs_path = [os.path.join(current_path, dir) for dir in items_path if os.path.isdir(os.path.join(current_path, dir))]

    for path in dirs_path:
        yield (path)


for dir_path in get_dirs():


    logger.info(f"____________________ path: {dir_path}")
    current_data_obj = data_class(dir_path)
    
    u = current_data_obj.reduce_dimensions_umap()
    current_data_obj.get_ideal_eps(0.3,10,0.05)
    # current_data_obj.perform_grid_search()

    # plt.scatter(u[:,0], u[:,1])
    # plt.show()

    # # find number of clusters:
    # best_numb_clusters = current_data_obj.find_best_num_clusters(min_n_clusters=2, max_n_clusters=10)
    # best_eps = current_data_obj.get_ideal_eps(min_eps=0.3, max_eps=10, step=0.05)
    # best_min_samples = current_data_obj.get_ideal_min_samples(min_min_samples=1, max_min_samples=15, step=1)
    # print("hey", current_data_obj.ideal_min_values)
    # current_data_obj.run_dbscan_over_umap(min_samples=best_min_samples[0], eps= best_eps)


    # # Plot the UMAP with colors based on DBSCAN clusters
    # plt.figure(figsize=(8, 6))
    # sns.scatterplot(x='UMAP_1', y='UMAP_2', hue='DBSCAN_Cluster', data=df, palette='viridis', legend='full')
    # plt.title('UMAP Clustering with DBSCAN')
    # plt.show()