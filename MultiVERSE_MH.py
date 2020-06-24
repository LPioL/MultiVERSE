



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
from evalne.evaluation.evaluator import LPEvaluator
from evalne.evaluation.score import Scoresheet
import functions as f
from sklearn.linear_model import LogisticRegressionCV 
from evalne.evaluation.split import EvalSplit
import pandas as pd
import networkx as nx


def main(args=None):
        
    parser = argparse.ArgumentParser(description='k-fold cross validation')
    parser.add_argument('-k', type=int, help='kfold')
    args = parser.parse_args(args)
    print(args)
    
    Test_networks = ['./Multiplex_Het/relabeled_curated_gene_disease_associations_MESH.csv',
                     './Multiplex_Het/relabeled_multiplex_disease.txt',
                     './Multiplex_Het/relabeled_multiplex_without_Coexp.edges'] 



    ########################################################################
    # Parameters multiverse and train/test
    ########################################################################

    EMBED_DIMENSION = 128
    CLOSEST_NODES = np.int64(300)
    NUM_SAMPLED = np.int64(10)
    LEARNING_RATE = np.float64(0.01)
    KL = False
    NB_CHUNK = np.int64(1)
    CHUNK_SIZE = np.int64(100)
    NUM_STEPS_1 = np.int64(100*10**6/CHUNK_SIZE)
    graph_name = 'test_disease_gene'
    
    ##################################################################################
    # !! Careful !! Check if nodes in the bipartite have the same nodes int he multiplex
    # networks
    ##################################################################################
    

    
    

    ###################################################################################"
    # MULTIVERSE
    ###################################################################################"
    r_readRDS = robjects.r['readRDS']
    
    print('RWR')
    proc = subprocess.Popen(['Rscript',  './Multiverse-master_MH/GenerateSimMatrix_MH.R', \
              '-n', '.'+ Test_networks[2],  \
              '-m', '.'+ Test_networks[1],  \
              '-b', '.' + Test_networks[1], 
              '-o', '../ResultsRWR/MatrixSimilarityMultiplexHet'+graph_name, '-c','40'])

    proc.wait()
    proc.kill()
    
    r_DistancematrixPPI = r_readRDS('./ResultsRWR/MatrixSimilarityMultiplexHet'+graph_name +'.rds') 

    import gc
    gc.collect()

        ########################################################################
        # Processing of the network
        ########################################################################
    reverse_data_DistancematrixPPI, list_neighbours, nodes, data_DistancematrixPPI, neighborhood, nodesstr \
     = f.netpreprocess_hetero(r_DistancematrixPPI, KL, CLOSEST_NODES)
     
    np.save('nodes', nodes)
    np.save('data', data_DistancematrixPPI)
    np.save('nodesstr_drug', nodesstr)


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
    np.save(str('embeddings'),embeddings)





if __name__ == "__main__":
    main()
    
    
    
  














