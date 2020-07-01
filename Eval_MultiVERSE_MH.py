



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
import functions_tricks_numba as fnumba
from sklearn.linear_model import LogisticRegressionCV 
from evalne.evaluation.split import EvalSplit
import pandas as pd
import networkx as nx


def main(args=None):
        
    
def main(args=None):
        
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
    KL = False
    NB_CHUNK = np.int64(1)
    CHUNK_SIZE = np.int64(100)
    NUM_STEPS_1 = np.int64(100*10**6/CHUNK_SIZE)

    train_frac =0.7
    solver = 'lbfgs'
    max_iter= 1000
    split_alg = 'random' # random naive fast spanning_tree
    lp_model = LogisticRegressionCV(Cs=10, cv= 5, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1.0, max_iter=max_iter, \
                        multi_class='ovr', n_jobs=42, random_state=None, refit=True, scoring='roc_auc', solver=solver, tol=0.0001, verbose=0) 
    
    # Load graph for EvalNE
    graph_name = 'LP_drug_gene'
    
    
    
    ###################################################################################"
    # MULTIVERSE
    ###################################################################################"
    r_readRDS = robjects.r['readRDS']
    
    print('RWR')
    proc = subprocess.Popen(['Rscript',  './Multiverse-master_MH/GenerateSimMatrix_MH.R', \
              '-n', '.'+ Networks_for_embedding[1],  \
              '-m', '.'+ Networks_for_embedding[2],  \
              '-b', '../Save_graphs/'+ 'heterogeneous_graph_' + 'processed' + '_'+ graph_name+'.txt', '-o', \
              '../ResultsRWR/MatrixSimilarityMultiplexHet'+graph_name, '-c','40'])

    proc.wait()

    
    proc.kill()

    print('bye')
    
    r_DistancematrixPPI = r_readRDS('./ResultsRWR/MatrixSimilarityMultiplexHet'+graph_name +'.rds') 

    import gc
    gc.collect()

        ########################################################################
        # Processing of the network
        ########################################################################
    reverse_data_DistancematrixPPI, list_neighbours, nodes, data_DistancematrixPPI, neighborhood, nodesstr \
     = fnumba.netpreprocess_hetero(r_DistancematrixPPI, KL, CLOSEST_NODES)
     
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
    
    embeddings = fnumba.train(neighborhood, nodes, list_neighbours, NUM_STEPS_1, NUM_SAMPLED, LEARNING_RATE, \
                         CLOSEST_NODES, CHUNK_SIZE, NB_CHUNK, embeddings, reverse_data_DistancematrixPPI)
    np.save(str('embeddings'),embeddings)
    date = datetime.datetime.now()
    os.rename('embeddings.npy', str('best_embeddings_Clustering'+'_'+graph_name+'_'+str(date)+'.npy'))
    os.replace(str('best_embeddings_Clustering'+'_'+graph_name+'_'+str(date)+'.npy'), './Save_embeddings/'+ str('best_embeddings_Clustering'+'_'+graph_name+'_'+str(date)+'.npy'))
    X = dict(zip(range(embeddings.shape[0]), embeddings))
    X = {str(int(nodesstr[key])+1): X[key] for key in X}


             
        ########################################################################
        # Link prediction for evaluation of MH
        ######################################################################## 

    edge_emb = ['hadamard', 'weighted_l1', 'weighted_l2', 'average', 'cosine']
    results_embeddings_methods = dict()

    for i in range (len(edge_emb)):
        tmp_result_multiverse = nee.evaluate_ne(data_split=nee.traintest_split, X=X, method="Multiverse", edge_embed_method=edge_emb[i], label_binarizer=lp_model)
        results_embeddings_methods[tmp_result_multiverse.method +'_'  + str(edge_emb[i])] = tmp_result_multiverse.get_all()[1][4]


    ########################################################################
    # Analysis and saving of the results
    ######################################################################## 
 
    Result_file_dict = 'Result_LinkpredMultiplexHet_dict'+graph_name+'_Multi_'+str(date)+'.txt'
    file = open(Result_file_dict,'w+')    
    file.write(str(results_embeddings_methods))
    file.close() 
    os.replace(Result_file_dict, './Save_results/' + Result_file_dict)
    
    Result_file = 'Result_LinkpredMultiplex_'+graph_name+'_Multi_'+split_alg+'_'+str(date)+'.txt'
    with open(Result_file,"w+") as overall_result:
       print("%s: \n\
                EMBED_DIMENSION: %s \n\
                CLOSEST_NODES: %s  \n\
                NUM_STEPS_1: %s  \n\
                NUM_SAMPLED: %s  \n\
                LEARNING_RATE: %s  \n\
                CHUNK_SIZE: %s  \n\
                NB_CHUNK: %s  \n\
                KL: %s \n\
                train_frac: %s \n\
                solver: %s \n\
                max_iter: %s  \n\
                split_alg: %s  \n\
                "% (str(date), EMBED_DIMENSION, CLOSEST_NODES, NUM_STEPS_1, NUM_SAMPLED, LEARNING_RATE, CHUNK_SIZE, NB_CHUNK, KL, train_frac, solver, max_iter, split_alg), file=overall_result)
             
       print('Overall MULTIVERSE AUC hadamard:', results_embeddings_methods['Multiverse_hadamard'], file=overall_result)
       print('Overall MULTIVERSE AUC weighted_l1:', results_embeddings_methods['Multiverse_weighted_l1'], file=overall_result)
       print('Overall MULTIVERSE AUC weighted_l2:', results_embeddings_methods['Multiverse_weighted_l2'], file=overall_result)
       print('Overall MULTIVERSE AUC average:', results_embeddings_methods['Multiverse_average'], file=overall_result)
       print('Overall MULTIVERSE AUC cosine:', results_embeddings_methods['Multiverse_cosine'], file=overall_result)
       
  
    overall_result.close() 
    os.replace(Result_file, './Save_results/'+ Result_file)
    
    print('End')

if __name__ == "__main__":
    main()
