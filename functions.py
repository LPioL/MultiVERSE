#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 08:20:28 2019

@author: Léo Pio-Lopez
"""

import numpy as np
import scipy as sc
from numba import jit, njit, prange, autojit
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
    
def jsd_compute(p, q, base=np.e):
    '''
        Implementation of pairwise `jsd` based on  
        https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
    '''
    ## convert to np.array
    p, q = np.asarray(p), np.asarray(q)
    ## normalize p, q to probabilities
    p, q = p/p.sum(), q/q.sum()
    m = 1./2*(p + q)
    return sc.stats.entropy(p,m, base=base)/2. + sc.stats.entropy(q, m, base=base)/2.
    
def jsd_matrix(data):
    jsd_matrix = np.ones(np.shape(data))
    for i in range( np.shape(data)[0]):
        for j in range( np.shape(data)[0]):
            jsd_matrix[i,j] = jsd_compute(data[i,:], data[j,:])
    data = jsd_matrix 
    return data
        
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

# from evalNE
def eval_other(nee, edge_emb):
    """
    Experiment to test other embedding methods not integrated in the library.
    """
    print('Evaluating Embedding methods...')

    # Set edge embedding methods
    # Other options: 'weighted_l1', 'weighted_l2'
    edge_embedding_methods = edge_emb


    # Evaluate methods from OpenNE
    # Set the methods
    methods = ['node2vec', 
        'deepwalk', 
        #'GreRep'
        'line']

    # Set the commands
    commands = [
        'python -m openne --method node2vec --graph-format edgelist --number-walks 10 --walk-length 80 --workers 4',
        'python -m openne --method deepWalk --graph-format edgelist --number-walks 10 --walk-length 80 --workers 4',
        'python -m openne --method line --graph-format edgelist --epochs 100']
    
    # Set parameters to be tuned
    tune_params = ['--p 0.5 1 --q 0.5 1', None, None, None]
    
    # For each method evaluate
    for i in range(len(methods)):
        command = commands[i] + " --input {} --output {} --representation-size {}"
        nee.evaluate_cmd(method_name=methods[i], method_type='ne', command=command,
                         edge_embedding_methods=edge_embedding_methods, input_delim=' ', output_delim=' ',
                         tune_params=tune_params[i])

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

def netpreprocess(r_DistancematrixPPI, graph_path, KL, CLOSEST_NODES):
    
    # Number of nodes in the network and computation of neighborrhood
    rawdata_DistancematrixPPI = np.array(r_DistancematrixPPI)
    rawdata_DistancematrixPPI= np.transpose(rawdata_DistancematrixPPI)
    node_size = np.shape(rawdata_DistancematrixPPI)[0]
    neighborhood=[]
    for i in range(node_size):
        neighborhood.append(np.shape(np.extract(rawdata_DistancematrixPPI[i,:] > 1/node_size, rawdata_DistancematrixPPI[i,:]))[0])

    # If several components
    mini = []
    nodes_incomponent=[]
    G = loadGraphFromEdgeListTxt(graph_path, directed=False)
    nb_component= nx.number_connected_components(G)
    components=sorted(nx.connected_components(G), key = len, reverse=True)
    for i in range(1,nb_component):
        nodes_incomponent.append(list(components[i]))
    rawdata_DistancematrixPPI = np.array(r_DistancematrixPPI)
    
    # We set the diagonal to the min probability of the distribution as we don't want a high similarity of the node to itself
    component=[]
    for i in range( 0,np.shape(rawdata_DistancematrixPPI)[0]):
        mini.append(np.min(rawdata_DistancematrixPPI[i,:][np.nonzero(rawdata_DistancematrixPPI[i,:])]))
        rawdata_DistancematrixPPI[i,i]= mini[i]
        rawdata_DistancematrixPPI[i,component]=mini[i]    
    rawdata_DistancematrixPPI= np.transpose(rawdata_DistancematrixPPI)
        
    # Normalization 
    rawdata_DistancematrixPPI  = normalize(rawdata_DistancematrixPPI, axis=1, norm='l1')
    data_DistancematrixPPI = rawdata_DistancematrixPPI
     
    # Names of the nodes in the PPI network (a vocab in the sense of skipgram)
    nodes = list(r_DistancematrixPPI.colnames)
    nodes = [int(i) for i in nodes]
    if nodes[0] == 1:
        nodes = nodes - np.ones(len(nodes), dtype=int)
    nodesstr = [str(i) for i in nodes]
    
 
    list_neighbours = []
    reverse_data_DistancematrixPPI = []
    for i in range(node_size):
        sort_genes = data_DistancematrixPPI[i,:]
        sort_values = sorted(sort_genes, reverse=True)
        sort_genes = data_DistancematrixPPI[i,:]
        sort_genes = np.argsort(-sort_genes)
        
        reverse_data_DistancematrixPPI.append(sort_values)
        list_neighbours.append(sort_genes)
        
    reverse_data_DistancematrixPPI=np.asarray(reverse_data_DistancematrixPPI)
    list_neighbours = np.asarray(list_neighbours)
    
    if KL==True:
        list_neighbours = np.delete(list_neighbours, 0,1)
    
    reverse_data_DistancematrixPPI  = normalize(reverse_data_DistancematrixPPI[:,0:CLOSEST_NODES], axis=1, norm='l1')
    
    return reverse_data_DistancematrixPPI, list_neighbours, nodes, data_DistancematrixPPI,nodes_incomponent, neighborhood, nodesstr
     
def netpreprocess_hetero(r_DistancematrixPPI, KL, CLOSEST_NODES):
    
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

    
    # to keep there
    rawdata_DistancematrixPPI= np.transpose(rawdata_DistancematrixPPI)
    

    # Normalization 
    rawdata_DistancematrixPPI  = normalize(rawdata_DistancematrixPPI, axis=1, norm='l1')
    
    if KL == True:

        data_DistancematrixPPI = jsd_matrix(rawdata_DistancematrixPPI) 
        np.save('jsd_ppi_tricks', data_DistancematrixPPI)
        data_DistancematrixPPI=np.asarray(data_DistancematrixPPI)
        data_DistancematrixPPI= 1 - data_DistancematrixPPI
        data_DistancematrixPPI  = normalize(data_DistancematrixPPI, axis=1, norm='l1')   
    
    
    else:
       
        data_DistancematrixPPI= rawdata_DistancematrixPPI
    
    
    # Names of the nodes in the PPI network (a vocab in the sense of skipgram)
    nodes = list(r_DistancematrixPPI.colnames)
    nodes = [int(i) for i in nodes]
    if nodes[0] == 1:
        nodes = nodes - np.ones(len(nodes), dtype=int)
    nodesstr = [str(i) for i in nodes]
    
    # Context = WINDOW_SIZE nodes the most similar/closest
    DistancematrixPPI = pd.DataFrame(data_DistancematrixPPI, nodesstr, nodesstr )
    
    list_neighbours = []
    reverse_data_DistancematrixPPI = []
    for i in range(node_size):
        sort_genes =data_DistancematrixPPI[i,:]
        sort_values = sorted(sort_genes, reverse=True)
        sort_genes = np.argsort(-sort_genes)
        
        reverse_data_DistancematrixPPI.append(sort_values)
        list_neighbours.append(sort_genes)
        
    reverse_data_DistancematrixPPI=np.asarray(reverse_data_DistancematrixPPI)
    list_neighbours = np.asarray(list_neighbours)
    
    if KL==True:
        list_neighbours = np.delete(list_neighbours, 0,1)
    

    reverse_data_DistancematrixPPI  = reverse_data_DistancematrixPPI[:,0:CLOSEST_NODES]
    reverse_data_DistancematrixPPI  = normalize(reverse_data_DistancematrixPPI, axis=1, norm='l1')
    
    return reverse_data_DistancematrixPPI, list_neighbours, nodes, data_DistancematrixPPI, neighborhood, nodesstr
        
   

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
