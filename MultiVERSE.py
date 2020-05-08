# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 11:09:00 2019

@author: winkidogan
"""


import subprocess
import numpy as np
import argparse
from MNE_LP_v2 import load_network_data
import os
import datetime
import rpy2.robjects as robjects
import networkx as nx
from evalne.methods import similarity
from evalne.evaluation.evaluator import LPEvaluator
from evalne.evaluation.split import EvalSplit
from evalne.evaluation.score import Scoresheet
from openne.node2vec import Node2vec
from openne.line import LINE
from openne.graph import Graph as Gr
import functions as f
from sklearn.linear_model import LogisticRegressionCV 
import pandas as pd

import gc



def main(args=None):
               
    Test_networks = ['./Dataset/Lazega-Law-Firm_multiplex.edges']
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

    # LOAD GRAPH FOR EVALNE
    graph_name = os.path.basename(graph_path)  


 
    ###################################################################################"
    # MULTIVERSE
    ###################################################################################"
    r_readRDS = robjects.r['readRDS']
    
    proc = subprocess.Popen(['Rscript',  './RWR/GenerateSimMatrix.R', \
              '-n', '../Dataset/'+Test_networks[args.k], '-o', \
              '../ResultsRWR/MatrixSimilarityMultiplex'+graph_name, '-c','40'])
    
    proc = subprocess.Popen(['Rscript',  './RWR/GenerateSimMatrix_MH.R', \
              '-n', '../Dataset/'+Test_networks[args.k], '-o', \
              '../ResultsRWR/MatrixSimilarityMultiplex'+graph_name, '-c','40'])

    proc.wait() 
    pid = proc.pid 
    proc.kill()
    os.system('module unload R/3.4.0')

    r_DistancematrixPPI = r_readRDS('./ResultsRWR/MatrixSimilarityMultiplex'+graph_name +'.rds') 

    gc.collect()

        ########################################################################
        # Processing of the network
        ########################################################################
    reverse_data_DistancematrixPPI, list_neighbours, nodes, data_DistancematrixPPI, nodes_incomponent, nodesstr \
     = f.netpreprocess_optimized(r_DistancematrixPPI, graph_path, KL, CLOSEST_NODES)
     
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
    embeddings = functions.traintrain(nodes, list_neighbours, NUM_STEPS_1, NUM_SAMPLED, LEARNING_RATE, \
                         CLOSEST_NODES, CHUNK_SIZE, NB_CHUNK, embeddings, reverse_data_DistancematrixPPI)
    np.save(str('embeddings'),embeddings)

    print('End')

if __name__ == "__main__":
    main()
