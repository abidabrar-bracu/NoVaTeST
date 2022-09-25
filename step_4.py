import pandas as pd
import numpy as np
import pickle
from skimage.filters import threshold_otsu
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
import argparse


parser = argparse.ArgumentParser(description='Clusters the noisy genes based on variance pattern and saves the results in a npz file')
parser.add_argument('-i','--input', help='Path of the dictionary file created in step 3', type=str, required=True)
parser.add_argument('-o','--output', help='Path of the npz file to save the predicted_clusters and cluster representatives', type=str, required=True)
parser.add_argument('-N','--num', help='Number of clusters', type=int, required=True)


args = vars(parser.parse_args())


with open(args["input"], 'rb') as f:
    d =  pickle.load(f)


hgp_var_all = []
for g in d:
    xx = d[g]['var_hetero'] > threshold_otsu(d[g]['var_hetero'])
    hgp_var_all.append(xx)
hgp_var_all = np.array(hgp_var_all).squeeze()

jaccard_distances = pdist(hgp_var_all, metric='jaccard')
jaccard_distances = squareform(jaccard_distances)
jaccard_similarity = 1-jaccard_distances


cluster = AgglomerativeClustering(n_clusters=args["num"], affinity='precomputed', linkage="complete")  
predicted_clusters = cluster.fit_predict(jaccard_distances)

HGP_vars = []
for g in d:
    HGP_vars.append(d[g]['var_hetero'])
    
HGP_vars = np.array(HGP_vars)
cluster_representative = []

for i in np.unique(predicted_clusters):
    idx = np.where(predicted_clusters == i)
    y_avg_idx = HGP_vars[idx, :].mean(1)
    cluster_representative.append(y_avg_idx)

np.savez(args["output"], predicted_clusters=predicted_clusters, cluster_representative=cluster_representative)
