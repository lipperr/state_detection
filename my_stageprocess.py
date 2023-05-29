# Module for tools for process of finding stages

import numpy as np
import scipy as sp
import statistics, math
import random as rd
import pandas as pd
import matplotlib.pyplot as plt

from math import sqrt, ceil

from sklearn import metrics
from sklearn import decomposition, cluster
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from scipy.spatial.distance import correlation
from IPython.display import display


# Calculate variance of clusters
def cluster_variance(df_cluster):
    cl_center = df_cluster.mean().to_numpy()
    cl_variance = 0
    for i in range(len(df_cluster.index)):
        cl_element = df_cluster.iloc[[i]].to_numpy()
        dist_to_center = np.linalg.norm(cl_element - cl_center)
        cl_variance += dist_to_center**2
        
    return cl_variance

# Calculate Ward distance first way (through variance)
def clusters_dist2_ward(df_cluster1, df_cluster2):
    var_cl1 = cluster_variance(df_cluster1)
    var_cl2 = cluster_variance(df_cluster2)
    var_cl_both = cluster_variance(pd.concat([df_cluster1, df_cluster2]))
    
    return var_cl_both - var_cl1 - var_cl2

# Calculate Ward distance second way
def clusters_dist_ward(df_cluster1, df_cluster2):
    cl1_center = df_cluster1.mean().to_numpy()
    cl2_center = df_cluster2.mean().to_numpy()
    n1 = len(df_cluster1.index)
    n2 = len(df_cluster2.index)
    
    return (n1*n2/(n1 + n2))*np.linalg.norm(cl1_center - cl2_center)**2

# All pairwise distances between elements of two clusters
def clusters_pairwise_dist(df_cluster1, df_cluster2):
    n1 = len(df_cluster1.index)
    n2 = len(df_cluster2.index)
    cl_dist_matrix = np.zeros((n1, n2))
    
    for i in range(n1):
        for j in range(n2):
            cl1_elem = df_cluster1.iloc[i].to_numpy()
            cl2_elem = df_cluster2.iloc[j].to_numpy()
            cl_dist_matrix[i,j] = np.linalg.norm(cl1_elem - cl2_elem)
            
    return cl_dist_matrix

# Confidence interval for the median
def median_confidence_interval(dx, cutoff=.95):
    ''' cutoff is the significance level as a decimal between 0 and 1'''
    dx = dx.sort_values(ascending=True, ignore_index=True)
    factor = statistics.NormalDist().inv_cdf((1+cutoff)/2)
    factor *= math.sqrt(len(dx)) # avoid doing computation twice

    lix = round(0.5*(len(dx)-factor))
    uix = min(round(0.5*(1+len(dx)+factor)), len(dx)-1)

    return (dx[lix],dx[uix])

# Transforming clusters into stages
def form_stages(cl_labels):
    n_clusters = len(np.unique(cl_labels))
    cl_edges = []
# Form array of edges of the clusters
    for i in range(n_clusters):
        cl_samples = np.where(cl_labels == i)[0]
        #print('Cluster '+str(i+1)+': ', cl_samples[0], cl_samples[-1])
        cl_edges.append(cl_samples[0])
        cl_edges.append(cl_samples[-1] + 1)
    cl_edges = np.unique(cl_edges)
    #print(cl_edges)

    return cl_edges

# Form bands array and new_labels list for stages
def form_stage_bands(st_edges, n_samples):
    # Cluster labels for stages
    n_stages = len(st_edges)-1
    new_labels = np.zeros(n_samples)
    for _st in range(n_stages):
        for i in range(len(new_labels)):
            if (st_edges[_st] <= i < st_edges[_st+1]):
                new_labels[i] = _st
    
    # Forming stage bands list
    st_bands = []
    for i in range(n_stages):
        st_samples = np.where(new_labels == i)[0]
        st_bands.append((st_samples[0], st_samples[-1], 'St'+str(i+1)))

    return st_bands, new_labels

# Merge small stages with neighbours 
def merge_stages_1st_step(df_features, st_edges, len_threshold=60):
    
    if len(st_edges) <= 2: 
        return st_edges
    
    st_lengths = np.array([st_edges[i+1] - st_edges[i] for i in range(len(st_edges)-1)])
    st_min_len = st_lengths.min()
    st_min_len_ind = st_lengths.argmin()
    
    while (st_min_len <= len_threshold):
        if (st_min_len_ind == len(st_lengths)-1):
            st_edges = np.delete(st_edges, st_min_len_ind)
        elif (st_min_len_ind == 0):
            st_edges = np.delete(st_edges, st_min_len_ind+1)
        else:
            st_dist_left = clusters_dist_ward(df_features.iloc[st_edges[st_min_len_ind-1]:st_edges[st_min_len_ind]], 
                                              df_features.iloc[st_edges[st_min_len_ind]:st_edges[st_min_len_ind+1]])
            st_dist_right = clusters_dist_ward(df_features.iloc[st_edges[st_min_len_ind]:st_edges[st_min_len_ind+1]], 
                                               df_features.iloc[st_edges[st_min_len_ind+1]:st_edges[st_min_len_ind+2]])
            if st_dist_left <= st_dist_right:
                st_edges = np.delete(st_edges, st_min_len_ind)
            else:
                st_edges = np.delete(st_edges, st_min_len_ind + 1)

        st_lengths = np.array([st_edges[i+1] - st_edges[i] for i in range(len(st_edges)-1)])
        st_min_len = st_lengths.min()
        st_min_len_ind = st_lengths.argmin()
        
    return st_edges     

# Merge stages if length > n_stages
def merge_stages_2nd_step(df_features, st_edges, dist_threshold = 0.2): 
    
    if len(st_edges) <= 2: 
        return st_edges
    
    st_lengths = np.array([st_edges[i+1] - st_edges[i] for i in range(len(st_edges)-1)])
    st_dist_list = np.array([clusters_dist_ward(df_features.iloc[st_edges[i-1]:st_edges[i]], 
                                                df_features.iloc[st_edges[i]:st_edges[i+1]]) 
                            for i in range(1, len(st_edges)-1)])
    st_min_dist = st_dist_list.min()
    st_min_dist_ind = st_dist_list.argmin()

    #while (len(st_lengths) > n_stages):
    while (st_min_dist <= dist_threshold*np.mean(st_dist_list)):
        st_edges = np.delete(st_edges, st_min_dist_ind+1)
        st_lengths = np.array([st_edges[i+1] - st_edges[i] for i in range(len(st_edges)-1)])
 
        st_dist_list = np.array([clusters_dist_ward(df_features.iloc[st_edges[i-1]:st_edges[i]], 
                                                  df_features.iloc[st_edges[i]:st_edges[i+1]]) 
                               for i in range(1, len(st_edges)-1)])
        st_min_dist = st_dist_list.min()
        st_min_dist_ind = st_dist_list.argmin()
       
    return st_edges   

def plot_stages(st_edges, n_samples):
    
    st_bands, new_labels = form_stage_bands(st_edges, n_samples)
    
    # Plotting stages
    fig, ax = plt.subplots(figsize=(6,4))
    for (x_start, x_end, _st) in st_bands:
        x = np.arange(x_start, x_end+1)
        y = np.full(len(x), _st)
        event_label = 'N=%d' % (x_end-x_start+1)
        ax.plot(x, y, '.', linewidth=5, label=event_label)
    ax.set(xlabel='Epoches', ylabel='Stages')
    ax.legend()#bbox_to_anchor=(0.25, 1, 0.1, -0.1))
    ax.tick_params(axis='both', labelsize=11, direction='in')
    fig.suptitle('Meditation stages', fontsize=16)
    plt.savefig('Meditation stages.png')
    

# Calculating stage distances (Ward, Centroid)
def calc_stage_distances(df_features, st_edges):
# Ward distances
    st_dist_ward = np.array([clusters_dist_ward(df_features.iloc[st_edges[i-1]:st_edges[i]], 
                                                df_features.iloc[st_edges[i]:st_edges[i+1]]) 
                            for i in range(1, len(st_edges)-1)])
    #print('Ward distance:', [round(x,2) for x in st_dist_ward])

# Centroid distance
    st_dist_centr = np.array([np.linalg.norm(df_features.iloc[st_edges[i-1]:st_edges[i]].mean().to_numpy() - 
                                             df_features.iloc[st_edges[i]:st_edges[i+1]].mean().to_numpy()) 
                              for i in range(1, len(st_edges)-1)]) 
    #print('Centroid distance:', [round(x,2) for x in st_dist_centr])

    return st_dist_ward, st_dist_centr


# Calculating stage distances (Ward, Centroid, Average, Single and Complete linkage)
def calc_stage_dist_full(df_features, st_edges):
# Ward distances
    st_dist_ward = np.array([clusters_dist_ward(df_features.iloc[st_edges[i-1]:st_edges[i]], 
                                                df_features.iloc[st_edges[i]:st_edges[i+1]]) 
                            for i in range(1, len(st_edges)-1)])
    #print('Ward distance:', [round(x,2) for x in st_dist_ward])

# Centroid distance
    st_dist_centr = np.array([np.linalg.norm(df_features.iloc[st_edges[i-1]:st_edges[i]].mean().to_numpy() - 
                                             df_features.iloc[st_edges[i]:st_edges[i+1]].mean().to_numpy()) 
                              for i in range(1, len(st_edges)-1)]) 
    #print('Centroid distance:', [round(x,2) for x in st_dist_centr])

# Linkage distances
    cl_dist_matr_list = [clusters_pairwise_dist(df_features.iloc[st_edges[i-1]:st_edges[i]], 
                                                df_features.iloc[st_edges[i]:st_edges[i+1]]) 
                         for i in range(1, len(st_edges)-1)]
# Average linkage 
    st_dist_avg_link = np.array([cl_dist_matr_list[i].mean() for i in range(len(cl_dist_matr_list))])
    #print('Average linkage:', [round(x,2) for x in st_dist_avg_link])
# Complete linkage
    st_dist_compl_link = np.array([cl_dist_matr_list[i].max() for i in range(len(cl_dist_matr_list))])
    #print('Complete linkage:', [round(x,2) for x in st_dist_compl_link])
# Single linkage
    st_dist_sing_link = np.array([cl_dist_matr_list[i].min() for i in range(len(cl_dist_matr_list))])
    #print('Single linkage:', [round(x,2) for x in st_dist_sing_link])

    return st_dist_ward, st_dist_centr, st_dist_avg_link, st_dist_compl_link, st_dist_sing_link
 
    
# Plotting stage distances (Ward, Centroid)
def plot_stage_distances(df_features, st_edges):
    
# Calculating stage distances    
    st_dist_ward, st_dist_centr = calc_stage_distances(df_features, st_edges)

# Forming DataFrame with distance values    
    st_dist_names = ['St'+str(i)+'_St'+str(i+1) for i in range(1, len(st_dist_ward)+1)]
    df_st_distances = pd.DataFrame(columns=['Method'] + st_dist_names)

# Ward distances
    st_dist_ward_dict = dict([(st_dist_names[i], st_dist_ward[i]) for i in range(len(st_dist_ward))])
    new_row = {'Method': 'Ward distance'}
    new_row.update(st_dist_ward_dict)
    df_st_distances = df_st_distances.append(new_row, ignore_index = True)

# Centroid distance
    st_dist_centr_dict = dict([(st_dist_names[i], st_dist_centr[i]) for i in range(len(st_dist_centr))])
    new_row = {'Method': 'Centroid distance'}
    new_row.update(st_dist_centr_dict)
    df_st_distances = df_st_distances.append(new_row, ignore_index = True)
    
# Plotting stage distances
    fig, ax = plt.subplots(figsize=(7,4))
    x = st_dist_names
    y_ward = [_dist/np.max(st_dist_ward) for _dist in st_dist_ward]
    y_centr = [_dist/np.max(st_dist_centr) for _dist in st_dist_centr]
   
    ax.plot(x, y_ward, linestyle='--', marker='o', label='Ward distance')
    ax.plot(x, y_centr, linestyle='--', marker='o', label='Centroid distance')
    
    ax.legend()
    ax.set(xlabel='Pairs of stages', ylabel='% from max distance')
    ax.tick_params(axis='both', labelsize=11, direction='in')    
    fig.suptitle("Distances between stages", fontsize=16)
    plt.savefig('Distances between stages.png')

    return df_st_distances


# Plotting stage distances (Ward, Centroid, Average, Single and Complete linkage)
def plot_stage_dist_full(df_features, st_edges):
    
# Calculating stage distances    
    st_dist_ward, st_dist_centr, st_dist_avg_link, st_dist_compl_link, st_dist_sing_link = calc_stage_dist_full(df_features, st_edges)
    
# Forming DataFrame with distance values    

    st_dist_names = ['St'+str(i)+'_St'+str(i+1) for i in range(1, len(st_dist_ward)+1)]
    df_st_distances = pd.DataFrame(columns=['Method'] + st_dist_names)

# Ward distances
    st_dist_ward_dict = dict([(st_dist_names[i], st_dist_ward[i]) for i in range(len(st_dist_ward))])
    new_row = {'Method': 'Ward distance'}
    new_row.update(st_dist_ward_dict)
    df_st_distances = df_st_distances.append(new_row, ignore_index = True)

# Centroid distance
    st_dist_centr_dict = dict([(st_dist_names[i], st_dist_centr[i]) for i in range(len(st_dist_centr))])
    new_row = {'Method': 'Centroid distance'}
    new_row.update(st_dist_centr_dict)
    df_st_distances = df_st_distances.append(new_row, ignore_index = True)
    
# Average linkage
    st_dist_avg_dict = dict([(st_dist_names[i], st_dist_avg_link[i]) for i in range(len(st_dist_avg_link))])
    new_row = {'Method': 'Average linkage'}
    new_row.update(st_dist_avg_dict)
    df_st_distances = df_st_distances.append(new_row, ignore_index = True)

# Complete linkage
    st_dist_compl_dict = dict([(st_dist_names[i], st_dist_compl_link[i]) for i in range(len(st_dist_compl_link))])
    new_row = {'Method': 'Complete linkage'}
    new_row.update(st_dist_compl_dict)
    df_st_distances = df_st_distances.append(new_row, ignore_index = True)

# Single linkage
    st_dist_sing_dict = dict([(st_dist_names[i], st_dist_sing_link[i]) for i in range(len(st_dist_sing_link))])
    new_row = {'Method': 'Single linkage'}
    new_row.update(st_dist_sing_dict)
    df_st_distances = df_st_distances.append(new_row, ignore_index = True)
    
# Plotting stage distances
    fig, ax = plt.subplots(figsize=(7,4))
    x = st_dist_names
    y_ward = [_dist/np.max(st_dist_ward) for _dist in st_dist_ward]
    y_centr = [_dist/np.max(st_dist_centr) for _dist in st_dist_centr]
    y_avg_lk = [_dist/np.max(st_dist_avg_link) for _dist in st_dist_avg_link]
    y_compl_lk = [_dist/np.max(st_dist_compl_link) for _dist in st_dist_compl_link]
    y_sing_lk = [_dist/np.max(st_dist_sing_link) for _dist in st_dist_sing_link]
    
    ax.plot(x, y_ward, linestyle='--', marker='o', label='Ward distance')
    ax.plot(x, y_centr, linestyle='--', marker='o', label='Centroid distance')
    ax.plot(x, y_avg_lk, linestyle='--', marker='o', label='Average linkage')
    ax.plot(x, y_compl_lk, linestyle='--', marker='o', label='Complete linkage')
    ax.plot(x, y_sing_lk, linestyle='--', marker='o', label='Single linkage')
    
    ax.legend()
    ax.set(xlabel='Pairs of stages', ylabel='% from max distance')
    ax.tick_params(axis='both', labelsize=11, direction='in')    
    fig.suptitle("Distances between stages", fontsize=16)
    plt.savefig('Distances between stages.png')

    return df_st_distances


# Plotting staging process for presentation
def plot_staging_process(cluster_labels):
    
    # Cluster labels for stages
    n_clusters = len(np.unique(cluster_labels))
    n_samples = len(cluster_labels)
    
    # Forming stage bands list
    cl_bands = []
    cl_samples = []
    for i in range(n_clusters):
        cl_samples.append(np.where(cluster_labels == i)[0])
        cl_bands.append((cl_samples[i][0], cl_samples[i][-1], 'Cl'+str(i+1)))
    #print(cl_bands)  

    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(3*4, 3*1))
    #plt.subplots_adjust(left=0.02)

    
    # Plotting clusters
    for i in range(n_clusters):
        x = cl_samples[i]
        y = np.full(len(x), cl_bands[i][2])
        event_label = 'N=%d' % (len(x))
        axs[0].plot(x, y, '.', linewidth=5, label=event_label)
    axs[0].set(xlabel='Epoches', ylabel='Clusters')
    axs[0].tick_params(axis='both', labelsize=11, direction='in')

    # Plotting stages (1st step)
    st_edges = form_stages(cl_method.labels_)
    st_bands, new_labels = form_stage_bands(st_edges, n_samples)
  
    for (x_start, x_end, _st) in st_bands:
        x = np.arange(x_start, x_end+1)
        y = np.full(len(x), _st)
        event_label = 'N=%d' % (x_end-x_start+1)
        axs[1].plot(x, y, '.', linewidth=5, label=event_label)
    axs[1].set(xlabel='Epoches', ylabel='Stages')
    axs[1].tick_params(axis='both', labelsize=11, direction='in')

    # Plotting stages (2nd step)
    st_edges = merge_stages_1st_step(df_features, st_edges, len_threshold=40)    
    st_bands, new_labels = form_stage_bands(st_edges, n_samples)
  
    for (x_start, x_end, _st) in st_bands:
        x = np.arange(x_start, x_end+1)
        y = np.full(len(x), _st)
        event_label = 'N=%d' % (x_end-x_start+1)
        axs[2].plot(x, y, '.', linewidth=5, label=event_label)
    axs[2].set(xlabel='Epoches', ylabel='Stages')
    axs[2].tick_params(axis='both', labelsize=11, direction='in')

    # Plotting stages (3rd step)
    st_edges = merge_stages_2nd_step(df_features, st_edges, n_stages=8)    
    st_bands, new_labels = form_stage_bands(st_edges, n_samples)
  
    for (x_start, x_end, _st) in st_bands:
        x = np.arange(x_start, x_end+1)
        y = np.full(len(x), _st)
        event_label = 'N=%d' % (x_end-x_start+1)
        axs[3].plot(x, y, '.', linewidth=5, label=event_label)
    axs[3].set(xlabel='Epoches', ylabel='Stages')
    axs[3].tick_params(axis='both', labelsize=11, direction='in')

    plt.tight_layout()#rect=[0,0.09,1,1])
    #fig.suptitle('Process of forming stages', fontsize=16)
    plt.savefig('Staging process.png')
    

