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
import utils as f
from sklearn.linear_model import LogisticRegressionCV 
from evalne.evaluation.split import EvalSplit
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
    KL = False
    NB_CHUNK = np.int64(1)
    CHUNK_SIZE = np.int64(100)
    NUM_STEPS_1 = np.int64(100*10**6/CHUNK_SIZE)

    train_frac =0.7
    solver = 'lbfgs'
    max_iter= 1000
    split_alg = 'random'
    lp_model = RandomForestClassifier(n_estimators=400, criterion='gini', max_depth=10, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, \
                                                      max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True,\
                                                      oob_score=True, n_jobs=cpu_number, random_state=777, verbose=0, warm_start=False) 
    
    # Load graph for EvalNE
    graph_name = 'Test_Eval'
    
    
        
    ##################################################################################
    # !! Careful !! 
    # Check if nodes in the bipartite have the same nodes in the multiplex
    # networks. If not you have to remove the nodes in the multiplexes not included in the  
    # bipartites
    ##################################################################################
    
    ###################################################################################
    # EvalNE Link prediction processing
    ###################################################################################

    data_bipartite = pd.read_csv(args.b, delimiter = ' ', header = None) 

    G_hetereogeneous = f.preprocess(args.b, '.', ' ', False,  False, True)
    print('Preprocessing done')
    G_hetereogeneous_traintest_split = EvalSplit()
    G_hetereogeneous_traintest_split.compute_splits(G_hetereogeneous, split_alg=split_alg, train_frac=train_frac, owa=False)
    nee = LPEvaluator(G_hetereogeneous_traintest_split, dim=EMBED_DIMENSION, lp_model=lp_model)
    G_heterogeneous_split = (G_hetereogeneous_traintest_split.TG)
    print('Splitting done')

    # Write the multiplex training graph for multiverse in extended edgelist format 'layer n1 n2 weight'
    file_multi = open('heterogeneous_graph_' + 'processed' + '_'+ graph_name, 'w+')  
    tmp_array_het = []
    multiplex_het_relabelled = pd.read_csv(args.b, sep=' ', index_col=None, header=None)
    tmp_array_het = np.asarray(G_heterogeneous_split.edges)
    tmp_array_het = np.asarray(G_heterogeneous_split.edges)#multiplex_het_relabelled[0]

    for i in range(len(tmp_array_het[:,0])):
        if tmp_array_het[i,0] in list(multiplex_het_relabelled[1]):
            tmp = tmp_array_het[i,0]
            tmp_array_het[i,0] = tmp_array_het[i,1]
            tmp_array_het[i,1] = tmp

    tmp_array_het = np.hstack((tmp_array_het, np.ones((len(tmp_array_het),1))))
    tmp_array_het = np.hstack((np.ones((len(tmp_array_het),1)), tmp_array_het))
    tmp_array_het = np.vstack(tmp_array_het)
    tmp_array_het = np.int_(tmp_array_het)

    
    np.savetxt(file_multi, tmp_array_het, fmt='%s', delimiter=' ', newline=os.linesep)
    
    file_multi.close()
    os.replace('heterogeneous_graph_' + 'processed' + '_'+ graph_name, './Generated_graphs/'+ 'heterogeneous_graph_' + 'processed' + '_'+ graph_name+'.txt')
    
      
    ###################################################################################
    # MULTIVERSE
    ###################################################################################
    r_readRDS = robjects.r['readRDS']
    
    print('RWR-MH')
    proc = subprocess.Popen(['Rscript',  './RWR/GenerateSimMatrix_MH.R', \
              '-n', args.n,  \
              '-m', args.m,  \
              '-b', '../Generated_graphs/heterogeneous_graph_processed_'+ graph_name+'.txt', 
              '-o', '../ResultsRWR/MatrixSimilarityMultiplexHet'+graph_name, '-c', str(cpu_number)])

    proc.wait() 
    proc.kill()
    print('RWR done')
    
    r_DistancematrixPPI = r_readRDS('./ResultsRWR/MatrixSimilarityMultiplexHet'+graph_name +'.rds') 

    import gc
    gc.collect()

        ########################################################################
        # Processing of the network
        ########################################################################
    reverse_data_DistancematrixPPI, list_neighbours, nodes, data_DistancematrixPPI, neighborhood, nodesstr \
     = f.netpreprocess_hetero(r_DistancematrixPPI, KL, CLOSEST_NODES)
    

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
    np.save('embeddings_MH', X)
    date = datetime.datetime.now()
    os.replace('embeddings_MH.npy', './ResultsMultiVERSE/'+ 'embeddings_MH.npy')

             
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
    
    Result_file = 'Result_LinkpredMultiplexHet_'+graph_name+'_'+str(date)+'.txt'
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
    os.replace(Result_file, './ResultsMultiVERSE/'+ Result_file)
    
    print('End')

if __name__ == "__main__":
    main()
