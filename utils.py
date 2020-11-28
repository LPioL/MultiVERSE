#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 08:20:28 2019

@author: Léo Pio-Lopez
"""

import numpy as np
import scipy as sc
from numba import jit, njit, prange
import math
from numpy.random import choice
from sklearn.preprocessing import normalize
import pandas as pd
import networkx as nx
from evalne.utils import preprocess as pp

@njit
def rand_choice_nb(arr, prob):
    """
    :param arr: A 1D numpy array of values to sample from.
    :param prob: A 1D numpy array of probabilities for the given samples.
    :return: A random sample from the given array with a given probability.
    """
    return arr[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]

@njit
def node_positive_weighted (u, list_neighbours, CLOSEST_NODES, reverse_data_DistancematrixPPI):
    if np.sum(reverse_data_DistancematrixPPI[u])==0:
        return int(list_neighbours[u][np.random.randint(1,CLOSEST_NODES)])
    else:
       probas = reverse_data_DistancematrixPPI[u,0:CLOSEST_NODES]
       draw = rand_choice_nb(list_neighbours[u,0:CLOSEST_NODES], probas)
       return int(draw)
            
@njit
def sigmoid(x): 
    return 1 / (1 + math.exp(-x))

def node_negative (u, list_neighbours, CLOSEST_NODES): 
    return int(list_neighbours[u][np.random.randint(np.size(list_neighbours[0])-CLOSEST_NODES,np.size(list_neighbours[0]))])

@njit 
def update (W_u, W_v, D, learning_rate, bias):
    sim = sigmoid(np.dot(W_u,W_v)-bias)
    gradient = (D - sim)*learning_rate
    W_u = W_u + gradient * W_v
    W_v = W_v + gradient * W_u
    return W_u, W_v, gradient

@njit(parallel=True)
def train(neighborhood, nodes, list_neighbours, NUM_STEPS, NUM_SAMPLED, LEARNING_RATE, CLOSEST_NODES, CHUNK_SIZE, NB_CHUNK, embeddings, reverse_data_DistancematrixPPI):
    
    nb_nodes=np.int64(np.shape(nodes)[0])
    # NCE biases
    nce_bias = np.float64(np.log(nb_nodes))
    nce_bias_neg = np.float64(np.log(nb_nodes/NUM_SAMPLED))
          
    for k in prange (NUM_STEPS):
        nodes_opt= np.random.randint(0, np.int64(nb_nodes),CHUNK_SIZE)   
        if k % int(10**6/CHUNK_SIZE) == 0:
            print (k)
            
        for i in prange (CHUNK_SIZE):
            u = nodes_opt[i]
            v = node_positive_weighted (u, list_neighbours, CLOSEST_NODES, reverse_data_DistancematrixPPI)
            embeddings[u,:], embeddings[v,:], gradientpos = update (embeddings[u,:],  embeddings[v,:], 1, LEARNING_RATE, nce_bias)

            for j in range (NUM_SAMPLED):
                v_neg =  list_neighbours[u, np.random.randint(CLOSEST_NODES+1,nb_nodes-1)]        
                embeddings[u,:], embeddings[v_neg,:], gradientneg = update (embeddings[u,:],  embeddings[v_neg,:], 0, LEARNING_RATE, nce_bias_neg)

    return embeddings

def knbrs(G, start, k):
    nbrs = set([start])
    for l in range(k):
        nbrs = set((nbr for n in nbrs for nbr in G[n]))
    return nbrs
     
# from EvalNE  
def preprocess(inpath, outpath, delimiter, directed, relabel, del_self_loops):
    """
    Graph preprocessing routine.
    """
    print('Preprocessing graph...')

    # Load a graph
    G = pp.load_graph(inpath, delimiter=delimiter, comments='#', directed=directed)

    # Preprocess the graph
    G, ids = pp.prep_graph(G, relabel=relabel, del_self_loops=del_self_loops)

    # Store preprocessed graph to a file
    pp.save_graph(G, output_path=outpath + "prep_graph.edgelist", delimiter=' ', write_stats=False)

    # Return the preprocessed graph
    return G

# from evalNE
def eval_baselines(nee, directed, scoresheet):
    """
    Experiment to test the baselines.
    """
    print('Evaluating baselines...')

    # Set the baselines
    methods = ['common_neighbours', 'jaccard_coefficient', 'adamic_adar_index',
               'preferential_attachment']

    # Evaluate baseline methods
    for method in methods:
        if directed:
            result = nee.evaluate_baseline(method=method, neighbourhood="in")
            scoresheet.log_results(result)
            result = nee.evaluate_baseline(method=method, neighbourhood="out")
            scoresheet.log_results(result)
        else:
            result = nee.evaluate_baseline(method=method)
            scoresheet.log_results(result)


# from MNE
def load_network_data(f_name):
    # This function is used to load multiplex data
    print('We are loading data from:', f_name)
    edge_data_by_type = dict()
    all_edges = list()
    all_nodes = list()
    with open(f_name, 'r') as f:
        for line in f:
            words = line[:-1].split(' ')
            if words[0] not in edge_data_by_type:
                edge_data_by_type[words[0]] = list()
            edge_data_by_type[words[0]].append((words[1], words[2]))
            all_edges.append((words[1], words[2]))
            all_nodes.append(words[1])
            all_nodes.append(words[2])
    all_nodes = list(set(all_nodes))
    # create common layer.
    all_edges = list(set(all_edges))
    edge_data_by_type['Base'] = all_edges
    print('Finish loading data')
    return edge_data_by_type, all_edges, all_nodes

def get_scores(nee, res, results):
    # Check the results
    results = nee.get_results()
#    for i in range (len(result_multiverse)):
#        results.append(result_multiverse[i])
    names = list()

    for i in range(len(results)):

        # Update the res variable with the results of the current repeat
        if len(res) != len(results):
            res.append(results[i].get_all())
        else:
            aux = results[i].get_all()
            res[i] = (res[i][0], [res[i][1][k] + aux[1][k] for k in range(len(aux[1]))])

        # Add the method names to a list
        if 'edge_embed_method' in results[i].params:
            names.append(results[i].method + '-' + results[i].params['edge_embed_method'])
        else:
            names.append(results[i].method)

    return names, res


     
def netpreprocess(r_DistancematrixPPI, CLOSEST_NODES):
    
    # Number of nodes in the network and computation of neighborrhood
    rawdata_DistancematrixPPI = np.array(r_DistancematrixPPI)
    rawdata_DistancematrixPPI= np.transpose(rawdata_DistancematrixPPI)
    node_size = np.shape(rawdata_DistancematrixPPI)[0]
    neighborhood=[]
    for i in range(node_size):
        neighborhood.append(np.shape(np.extract(rawdata_DistancematrixPPI[i,:] > 1/node_size, rawdata_DistancematrixPPI[i,:]))[0])
 

    # If several components
    mini = []
    rawdata_DistancematrixPPI = np.array(r_DistancematrixPPI)
    
    # change the diagonal and keep track of nodes of other components
    component=[]
    for i in range( 0,np.shape(rawdata_DistancematrixPPI)[0]):
        mini.append(np.min(rawdata_DistancematrixPPI[i,:][np.nonzero(rawdata_DistancematrixPPI[i,:])]))
        rawdata_DistancematrixPPI[i,i]= mini[i]
        rawdata_DistancematrixPPI[i,component]=mini[i]
    rawdata_DistancematrixPPI= np.transpose(rawdata_DistancematrixPPI)
    
    # Normalization 
    rawdata_DistancematrixPPI  = normalize(rawdata_DistancematrixPPI, axis=1, norm='l1')
    #data_DistancematrixPPI= rawdata_DistancematrixPPI
    
    
    # Names of the nodes in the PPI network (a vocab in the sense of skipgram)
    nodes = list(r_DistancematrixPPI.colnames)
    nodes = [int(i) for i in nodes]
    if nodes[0] == 1:
        nodes = nodes - np.ones(len(nodes), dtype=int)
    nodesstr = [str(i) for i in nodes]
  
    #DistancematrixPPI = pd.DataFrame(rawdata_DistancematrixPPI, nodesstr, nodesstr )
    list_neighbours = []
    reverse_data_DistancematrixPPI = []
    for i in range(node_size):
        sort_genes = rawdata_DistancematrixPPI[i,:]
        sort_values = sorted(sort_genes, reverse=True)
        sort_genes = np.argsort(-sort_genes)
        
        reverse_data_DistancematrixPPI.append(sort_values[0:CLOSEST_NODES])
        list_neighbours.append(sort_genes)
        
    reverse_data_DistancematrixPPI=np.asarray(reverse_data_DistancematrixPPI)
    list_neighbours = np.asarray(list_neighbours)
    
    
    #reverse_data_DistancematrixPPI  = reverse_data_DistancematrixPPI[:,0:CLOSEST_NODES]
    reverse_data_DistancematrixPPI  = normalize(reverse_data_DistancematrixPPI, axis=1, norm='l1')
    
    return reverse_data_DistancematrixPPI, list_neighbours, nodes, rawdata_DistancematrixPPI, neighborhood, nodesstr
        
   

def loadGraphFromEdgeListTxt(graph, directed=True):
    with open(graph, 'r') as g:
        if directed:
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        for lines in g:
            edge = lines.strip().split()
            if len(edge) == 3:
                w = float(edge[2])
            else:
                w = 1.0
            G.add_edge(int(edge[0]), int(edge[1]), weight=w)
    return G
    

    
        
