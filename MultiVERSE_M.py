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
import gc
import multiprocessing




def main(args=None):
          
    cpu_number = multiprocessing.cpu_count()     
    parser = argparse.ArgumentParser(description='Position of the test networks')
    parser.add_argument('-k', type=int, help='Test network')
    args = parser.parse_args(args)
    Test_networks = ['./Dataset/Multiplex/CKM-Physicians-Innovation_multiplex.edges']
    graph_path = Test_networks[args.k]
    
    ########################################################################
    # Parameters 
    ########################################################################
       
    EMBED_DIMENSION = 128   
    CLOSEST_NODES = np.int64(20)
    NUM_SAMPLED = np.int64(3)
    LEARNING_RATE = np.float64(0.01)
    KL = False
    NB_CHUNK = np.int64(1)
    CHUNK_SIZE = np.int64(10)
    NUM_STEPS_1 = np.int64(100*10**6/CHUNK_SIZE)
    graph_name = os.path.basename(graph_path)  


 
    ###################################################################################"
    # MULTIVERSE-M
    ###################################################################################"
    r_readRDS = robjects.r['readRDS']
    
    print('RWR-M')
    proc = subprocess.Popen(['Rscript',  './RWR/GenerateSimMatrix.R', \
              '-n', "."+Test_networks[args.k], '-o', \
              '../ResultsRWR/MatrixSimilarityMultiplex'+graph_name, '-c',str(cpu_number)])

    proc.wait() 
    pid = proc.pid 
    proc.kill()

    print('RWR done')
    
    r_DistancematrixPPI = r_readRDS('./ResultsRWR/MatrixSimilarityMultiplex'+graph_name +'.rds') 

    gc.collect()

        ########################################################################
        # Processing of the network
        ########################################################################
    reverse_data_DistancematrixPPI, list_neighbours, nodes, data_DistancematrixPPI, nodes_incomponent, neighborhood, nodesstr \
     = f.netpreprocess(r_DistancematrixPPI, graph_path, KL, CLOSEST_NODES)
     
    np.save('nodes', nodes)
    np.save('data', data_DistancematrixPPI)
    np.save('nodesstr', nodesstr)

        ########################################################################
        # Initialization
        ######################################################################## 
    embeddings = np.random.normal(0, 1, [np.size(nodes), EMBED_DIMENSION])

             
        ########################################################################
        # Training and saving best embeddings   
        ######################################################################## 

    nodes= np.asarray(nodes)
    embeddings = f.train(neighborhood, nodes, list_neighbours, NUM_STEPS_1, NUM_SAMPLED, LEARNING_RATE, \
                         CLOSEST_NODES, CHUNK_SIZE, NB_CHUNK, embeddings, reverse_data_DistancematrixPPI)

    X = dict(zip(range(embeddings.shape[0]), embeddings))
    X = {str(int(nodesstr[key])+1): X[key] for key in X}
    np.save('embeddings_M',X)
    date = datetime.datetime.now()
    os.replace('embeddings_M.npy', './ResultsMultiVERSE/'+ 'embeddings_MH.npy')

    print('End')



if __name__ == "__main__":
    main()
