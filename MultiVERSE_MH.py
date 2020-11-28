# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 11:09:00 2019

@author: Léo Pio-Lopez
"""

import subprocess
import numpy as np
import argparse
import os
import datetime
import rpy2.robjects as robjects
import networkx as nx
import utils as f
import pandas as pd
import networkx as nx
import multiprocessing

def main(args=None):
        
    cpu_number = multiprocessing.cpu_count()     
    parser = argparse.ArgumentParser(description='Path of networks')
    parser.add_argument('-n', type=str, help='Multiplex 1')
    parser.add_argument('-m', type=str, help='Multiplex 2')    
    parser.add_argument('-b', type=str, help='Bipartite')        
    
    args = parser.parse_args(args)
    print(args)
    
    Test_networks = ['./Multiplex_Het/heterogeneous_graph.txt',
                     './Multiplex_Het/Multiplex_1.txt',
                     './Multiplex_Het/Multiplex_2.txt'] 



    ########################################################################
    # Parameters multiverse and train/test
    ########################################################################

    EMBED_DIMENSION = 128
    CLOSEST_NODES = np.int64(300)
    NUM_SAMPLED = np.int64(10)
    LEARNING_RATE = np.float64(0.01)
    NB_CHUNK = np.int64(1)
    CHUNK_SIZE = np.int64(100)
    NUM_STEPS_1 = np.int64(100*10**6/CHUNK_SIZE)
    graph_name = 'test_MH'
    

    #If toy example
    EMBED_DIMENSION = 128
    CLOSEST_NODES = np.int64(2)
    NUM_SAMPLED = np.int64(10)
    LEARNING_RATE = np.float64(0.01)
    NB_CHUNK = np.int64(1)
    CHUNK_SIZE = np.int64(2)
    NUM_STEPS_1 = np.int64(100*10**6/CHUNK_SIZE)
    
    
    ##################################################################################
    # !! Careful !! 
    # Check if nodes in the bipartite have the same nodes in the multiplex
    # networks. If not you have to remove the nodes in the multiplexes not included in the  
    # bipartites
    ##################################################################################
    

    ###################################################################################"
    # MULTIVERSE-MH
    ###################################################################################"
    r_readRDS = robjects.r['readRDS']
    
    print('RWR-MH')
    proc = subprocess.Popen(['Rscript',  './RWR/GenerateSimMatrix_MH.R', \
              '-n', '.' + args.n,  \
              '-m', '.' + args.m,  \
              '-b', '.' + args.b, 
              '-o', '../ResultsRWR/MatrixSimilarityMultiplexHet' + graph_name, '-c', str(cpu_number)])

    proc.wait()
    proc.kill()
    print('RWR done')
      
    r_DistancematrixPPI = r_readRDS('./ResultsRWR/MatrixSimilarityMultiplexHet'+graph_name +'.rds') 

    import gc
    gc.collect()

        ########################################################################
        # Processing of the network
        ########################################################################
        
    reverse_data_DistancematrixPPI, list_neighbours, nodes, rawdata_DistancematrixPPI, neighborhood, nodesstr = f.netpreprocess(r_DistancematrixPPI, CLOSEST_NODES)
     
        ########################################################################
        # Initialization
        ######################################################################## 

    embeddings = np.random.normal(0, 1, [np.size(nodes), EMBED_DIMENSION])
 
        ########################################################################
        # Training and saving best embeddings   
        ######################################################################## 
    # Train and test during training
    neighborhood = np.asarray(neighborhood)
    nodes= np.asarray(nodes)
    
    embeddings = f.train(neighborhood, nodes, list_neighbours, NUM_STEPS_1, NUM_SAMPLED, LEARNING_RATE, \
                         CLOSEST_NODES, CHUNK_SIZE, NB_CHUNK, embeddings, reverse_data_DistancematrixPPI)
 
    X = dict(zip(range(embeddings.shape[0]), embeddings))
    X = {str(int(nodesstr[key])+1): X[key] for key in X}
    np.save('embeddings_MH',X)
    date = datetime.datetime.now()
    os.replace('embeddings_MH.npy', './ResultsMultiVERSE/'+ 'embeddings_MH.npy')
    
if __name__ == "__main__":
    main()
    
    
    
  













