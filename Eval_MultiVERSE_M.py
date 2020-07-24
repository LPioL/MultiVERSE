# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 11:09:00 2019

@author: Léo Pio-Lopez
"""

from sklearn.metrics import roc_auc_score
import math
import subprocess
import Node2Vec_LayerSelect
import numpy as np
import argparse
from MNE_LP_v2 import load_network_data
import sys
import os
import datetime
from operator import itemgetter
import rpy2.robjects as robjects
import argparse
import networkx as nx
from evalne.methods import similarity
from evalne.evaluation.evaluator import LPEvaluator
from evalne.evaluation.split import EvalSplit
from evalne.evaluation.score import Scoresheet
from evalne.utils import preprocess as pp
from openne.node2vec import Node2vec
from openne.line import LINE
from openne.graph import Graph as Gr
import shutil
import functions_tricks_numba as fnumba
from sklearn.linear_model import LogisticRegressionCV 
import pandas as pd
from evalne.evaluation.score import Scoresheet
from sklearn.svm import SVC 
import multiprocessing




def main(args=None):
          
    cpu_number = multiprocessing.cpu_count()     
    parser = argparse.ArgumentParser(description='Position of the test networks')
    parser.add_argument('-k', type=int, help='Test network')
    args = parser.parse_args(args)
    Test_networks = ['./Dataset/Multiplex/CKM-Physicians-Innovation_multiplex.edges']
    graph_path = Test_networks[args.k]
    
    ########################################################################
    # Parameters multiverse and train/test
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
    train_frac = 0.7
    solver = 'lbfgs'
    max_iter= 1000
    split_alg = 'spanning_tree' 
    
    lp_model = LogisticRegressionCV(Cs=10, cv= 5, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1.0, max_iter=max_iter, \
                        multi_class='ovr', n_jobs=42, random_state=None, refit=True, scoring='roc_auc', solver=solver, tol=0.0001, verbose=0) 

    edge_data_by_type, _, all_nodes = load_network_data(graph_path)
    nb_layers = len(edge_data_by_type.keys())

    # Divide multiplex graph in several in edgelist format
    for layer in range(nb_layers-1):
        file = open('multiplex_graph_layer_' + str(layer+1) + '_'+ graph_name, 'w+')  
        tmp_array = np.asarray(edge_data_by_type[str(layer +1)])
        np.savetxt(file, tmp_array, fmt='%s')
        file.close()
        os.replace('multiplex_graph_layer_' + str(layer+1) + '_'+ graph_name, 'Save_graphs/'+'multiplex_graph_layer_' + str(layer+1) + '_'+ graph_name)

    

    # Load each graph with EvalNE, preprocess and split train/test edges
    nee = list()
    G_original = list()
    Gsplit = list()
    traintestsplit = list()
    for layer in range(nb_layers-1):
        G_original.append(fnumba.preprocess('./Save_graphs/'+'multiplex_graph_layer_' + str(layer+1) + '_'+ graph_name, '.', ' ', directed = False,  relabel = False, del_self_loops = True))
        G_original_traintest_split = EvalSplit()
        G_original_traintest_split.compute_splits(G_original[layer], split_alg=split_alg, train_frac=train_frac, owa=False)
        traintestsplit.append(G_original_traintest_split)
        nee.append(LPEvaluator(G_original_traintest_split, dim=EMBED_DIMENSION, lp_model=lp_model))
        Gsplit.append(G_original_traintest_split.TG)

    
    # Write the multiplex training graph for multiverse in extended edgelist format 'layer n1 n2 weight'
    file_multi = open('multiverse_graph_' + 'training' + '_'+ graph_name, 'w+')  
    matrix_train_edges = []
    sorted_matrix_train_edges = []
    tmp_array_multi = []
    tmp_array_collapsed = []
    for layer in range(nb_layers-1):

        tmp_array = np.asarray(Gsplit[layer].edges)
        tmp_array = np.hstack((tmp_array, np.ones((len(tmp_array),1))))
        tmp_array = np.hstack(((layer+1)*np.ones((len(tmp_array),1)), tmp_array))
        tmp_array=np.vstack(tmp_array)
        tmp_array_multi.append(tmp_array)
               
        tmp_array_mat_train_edges = np.asarray(Gsplit[layer].edges)
        tmp_array_mat_train_edges = np.hstack((tmp_array_mat_train_edges,np.ones((len(tmp_array_mat_train_edges),1))))
        tmp_array_mat_train_edges = np.hstack(((layer)*np.ones((len(tmp_array),1)), tmp_array_mat_train_edges))
        matrix_train_edges.append(tmp_array_mat_train_edges)

        matrix_train_edges = sorted(tmp_array_mat_train_edges, key=itemgetter(1))
        sorted_matrix_train_edges.extend(matrix_train_edges)
        matrix_train_edges = []
           
    tmp_array_multi=np.vstack(tmp_array_multi)
    tmp_array_multi=np.int_(tmp_array_multi)
    np.savetxt(file_multi, tmp_array_multi, fmt='%s', delimiter=' ', newline=os.linesep)
    
    file_multi.close()
    os.replace('multiverse_graph_' + 'training' + '_'+ graph_name, './Save_graphs/'+ 'multiverse_graph_' + 'training' + '_'+ graph_name+'.txt')


    ###################################################################################"
    # MULTIVERSE
    ###################################################################################"
    r_readRDS = robjects.r['readRDS']
    
    proc = subprocess.Popen(['Rscript',  './Multiverse-master_MH/GenerateSimMatrix.R', \
              '-n', '../Save_graphs/'+'multiverse_graph_' + 'training' + '_'+ graph_name+'.txt', '-o', \
              '../ResultsRWR/MatrixSimilarityMultiplex'+graph_name, '-c','40'])

    proc.wait()
    pid = proc.pid 
    proc.kill()
    print('RWR done')
    r_DistancematrixPPI = r_readRDS('./ResultsRWR/MatrixSimilarityMultiplex'+graph_name +'.rds') 

        ########################################################################
        # Processing of the network
        ########################################################################
    reverse_data_DistancematrixPPI, list_neighbours, nodes, data_DistancematrixPPI, nodes_incomponent, nodesstr \
     = fnumba.netpreprocess_optimized(r_DistancematrixPPI, graph_path, KL, CLOSEST_NODES)

        ########################################################################
        # Initialization
        ######################################################################## 

    embeddings = np.random.normal(0, 1, [np.size(nodes), EMBED_DIMENSION])
         
        ########################################################################
        # Training and saving best embeddings   
        ######################################################################## 

    nodes= np.asarray(nodes)
    
    embeddings = fnumba.train_optimized(nodes, list_neighbours, NUM_STEPS_1, NUM_SAMPLED, LEARNING_RATE, \
                         CLOSEST_NODES, CHUNK_SIZE, NB_CHUNK, embeddings, reverse_data_DistancematrixPPI)
    np.save(str('embeddings'),embeddings)
    date = datetime.datetime.now()
    os.rename('embeddings.npy', str('best_embeddings_LP'+'_'+graph_name+'_'+str(date)+'.npy'))
    os.replace(str('best_embeddings_LP'+'_'+graph_name+'_'+str(date)+'.npy'), './Save_embeddings/'+ str('best_embeddings_LP'+'_'+graph_name+'_'+str(date)+'.npy'))
    X = dict(zip(range(embeddings.shape[0]), embeddings))
    X = {str(int(nodesstr[key])+1): X[key] for key in X}


    for layer in range(nb_layers-1):   
        for i in range (len(edge_emb)):
            tmp_result_multiverse = nee[layer].evaluate_ne(data_split=nee[layer].traintest_split, X=X, method="Multiverse", edge_embed_method=edge_emb[i],
            label_binarizer=lp_model)
            results_embeddings_methods[tmp_result_multiverse.method +'_' + str(layer) + str(edge_emb[i])] = tmp_result_multiverse.get_all()[1][4]

    
    ########################################################################
    # Analysis and saving of the results
    ######################################################################## 

    tmp_Multiverse_Result_hada = 0
    tmp_Multiverse_Result_wl1 = 0
    tmp_Multiverse_Result_wL2 = 0
    tmp_Multiverse_Result_avg = 0
    tmp_Multiverse_Result_cos = 0

    for layer in range(nb_layers-1):
        tmp_Multiverse_Result_hada += results_embeddings_methods['Multiverse'+'_'+str(layer) + str(edge_emb[0])]
        tmp_Multiverse_Result_wl1 += results_embeddings_methods['Multiverse'+'_'+str(layer) + str(edge_emb[1])]
        tmp_Multiverse_Result_wL2 += results_embeddings_methods['Multiverse'+'_'+str(layer) + str(edge_emb[2])]
        tmp_Multiverse_Result_avg += results_embeddings_methods['Multiverse'+'_'+str(layer) + str(edge_emb[3])]
        tmp_Multiverse_Result_cos += results_embeddings_methods['Multiverse'+'_'+str(layer) + str(edge_emb[4])]

    results_embeddings_methods['Multiverse_av_hadamard'] = tmp_Multiverse_Result_hada/(nb_layers-1)
    results_embeddings_methods['Multiverse_av_weighted_l1'] = tmp_Multiverse_Result_wl1/(nb_layers-1)
    results_embeddings_methods['Multiverse_av_weighted_l2'] = tmp_Multiverse_Result_wL2/(nb_layers-1)
    results_embeddings_methods['Multiverse_av_average'] = tmp_Multiverse_Result_avg/(nb_layers-1)
    results_embeddings_methods['Multiverse_av_cosine'] = tmp_Multiverse_Result_cos/(nb_layers-1)
   
 
    Result_file_dict = 'Result_LinkpredMultiplex_dict'+graph_name+'_Multi_'+str(date)+'.txt'
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
             
       print('Overall MULTIVERSE AUC hadamard:', results_embeddings_methods['Multiverse_av_hadamard'], file=overall_result)
       print('Overall MULTIVERSE AUC weighted_l1:', results_embeddings_methods['Multiverse_av_weighted_l1'], file=overall_result)
       print('Overall MULTIVERSE AUC weighted_l2:', results_embeddings_methods['Multiverse_av_weighted_l2'], file=overall_result)
       print('Overall MULTIVERSE AUC average:', results_embeddings_methods['Multiverse_av_average'], file=overall_result)
       print('Overall MULTIVERSE AUC cosine:', results_embeddings_methods['Multiverse_av_cosine'], file=overall_result)
            

    overall_result.close() 
    os.replace(Result_file, './Save_results/'+ Result_file)
    
    print('End')

if __name__ == "__main__":
    main()
