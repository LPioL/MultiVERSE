# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 11:09:00 2019

@author: Léo Pio-Lopez
"""

import subprocess
import numpy as np
import argparse
from MNE_LP_v2 import load_network_data
import os
import datetime
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
import functions_tricks_numba as fnumba
from sklearn.linear_model import LogisticRegressionCV 
import pandas as pd
from evalne.evaluation.score import Scoresheet



def main(args=None):
        
        parser = argparse.ArgumentParser(description='k-fold cross validation')
    parser.add_argument('-k', type=int, help='kfold')
    args = parser.parse_args(args)
    print(args)
    
    # We can add as many as multiplex we need and parallelize their execution for comparisons
    Test_networks = ['./Networks_Multiplex/Lazega-Law-Firm_Multiplex_Social/Dataset/Lazega-Law-Firm_multiplex.edges']
    graph_path = Test_networks[args.k]
    
    ########################################################################
    # Parameters multiverse and train/test
    ########################################################################
    
    if args.k == 0 :
        CLOSEST_NODES = np.int64(20)
        NUM_SAMPLED = np.int64(3)
        LEARNING_RATE = np.float64(0.01)
        KL = False
        NB_CHUNK = np.int64(1)
        CHUNK_SIZE = np.int64(10)
        NUM_STEPS_1 = np.int64(100*10**6/CHUNK_SIZE)
        
    EMBED_DIMENSION = 128   
    train_frac = 0.7
    solver = 'lbfgs'
    max_iter= 1000
    split_alg = 'spanning_tree' 
    lp_model = LogisticRegressionCV(Cs=10, cv= 5, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1.0, max_iter=max_iter, \
                        multi_class='ovr', n_jobs=42, random_state=None, refit=True, scoring='roc_auc', solver=solver, tol=0.0001, verbose=0) 

    # Load graph for EvalNE
    graph_name = os.path.basename(graph_path)
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
        #G = pp.load_graph('multiplex_graph_layer_' + str(layer+1) + '_'+ graph_name, '.', ' ', False)
        G_original.append(fnumba.preprocess('./Save_graphs/'+'multiplex_graph_layer_' + str(layer+1) + '_'+ graph_name, '.', ' ', directed = False,  relabel = False, del_self_loops = True))
        G_original_traintest_split = EvalSplit()
        G_original_traintest_split.compute_splits(G_original[layer], split_alg=split_alg, train_frac=train_frac, owa=False)
        traintestsplit.append(G_original_traintest_split)
        nee.append(LPEvaluator(G_original_traintest_split, dim=EMBED_DIMENSION, lp_model=lp_model))
        Gsplit.append(G_original_traintest_split.TG)

   
    file_collpased_original = open('collapsed_original' + '_'+ graph_name, 'w+')  
    tmp_array=[]
    for layer in range(nb_layers-1):        
        tmp_array.append( np.asarray(G_original[layer].edges))
    tmp_array=np.vstack(tmp_array)
    tmp_array=pd.DataFrame(tmp_array)
    tmp_array.drop_duplicates()
    tmp_array = np.asarray(tmp_array)
    np.savetxt(file_collpased_original, tmp_array, fmt='%s')
    file_collpased_original.close()
    
    # Write the multiplex training graph for multiverse in extended edgelist format 'layer n1 n2 weight' and for ohmnet 3 graph 
    #in normal edgelist format and we write the other graph in proper format for the different methods
    from operator import itemgetter
    file_multi = open('multiverse_graph_' + 'training' + '_'+ graph_name, 'w+')  
    file_collapsed_net =  open('collapsed_graph_' + 'training' + '_'+ graph_name, 'w+')  
    matrix_train_edges = []
    sorted_matrix_train_edges = []
    tmp_array_multi = []
    tmp_array_collapsed = []
    for layer in range(nb_layers-1):
        file_ohmnet = open('ohmnet'+'_graph_' + 'training' + '_' + graph_name + str(layer+1) , 'w+')
        
        tmp_array = np.asarray(Gsplit[layer].edges)
        tmp_array = np.hstack((tmp_array, np.ones((len(tmp_array),1))))
        tmp_array = np.hstack(((layer+1)*np.ones((len(tmp_array),1)), tmp_array))
        tmp_array=np.vstack(tmp_array)
        tmp_array_multi.append(tmp_array)
        
        tmp_array_coll = np.asarray(Gsplit[layer].edges)
        tmp_array_coll=np.vstack(tmp_array_coll)
        tmp_array_collapsed.append(tmp_array_coll)
        
        tmp_array_ohm = np.asarray(Gsplit[layer].edges)
        np.savetxt(file_ohmnet, tmp_array_ohm, fmt='%s', delimiter = ' ') 

        tmp_array_mat_train_edges = np.asarray(Gsplit[layer].edges)
        tmp_array_mat_train_edges = np.hstack((tmp_array_mat_train_edges,np.ones((len(tmp_array_mat_train_edges),1))))
        tmp_array_mat_train_edges = np.hstack(((layer)*np.ones((len(tmp_array),1)), tmp_array_mat_train_edges))
        matrix_train_edges.append(tmp_array_mat_train_edges)

        matrix_train_edges = sorted(tmp_array_mat_train_edges, key=itemgetter(1))
        sorted_matrix_train_edges.extend(matrix_train_edges)
        matrix_train_edges = []
        
        file_ohmnet.close()
        os.replace('ohmnet'+'_graph_' + 'training' + '_' + graph_name + str(layer+1), './Save_graphs/' + 'ohmnet'+'_graph_' + 'training' + '_' + graph_name + str(layer+1))
    
    tmp_array_multi=np.vstack(tmp_array_multi)
    tmp_array_multi=np.int_(tmp_array_multi)
    np.savetxt(file_multi, tmp_array_multi, fmt='%s', delimiter=' ', newline=os.linesep)
    
    tmp_array_collapsed=np.vstack(tmp_array_collapsed)
    tmp_array_collapsed=np.int_(tmp_array_collapsed)
    np.savetxt(file_collapsed_net, tmp_array_collapsed, fmt='%.18s', delimiter=' ', newline=os.linesep)

    file_multi.close()
    file_collapsed_net.close()
    os.replace('multiverse_graph_' + 'training' + '_'+ graph_name, './Save_graphs/'+ 'multiverse_graph_' + 'training' + '_'+ graph_name+'.txt')
    os.replace('collapsed_graph_' + 'training' + '_'+ graph_name, './Save_graphs/' + 'collapsed_graph_' + 'training' + '_'+ graph_name)

    # Ohmnet list and hierarchy
    file_list_ohmnet = open('ohmnet_graphlist' + 'training' + '_'+ graph_name + '.list', 'w+') 
    file_hierarchy_ohmnet = open('ohmnet_hierarchy_' + 'training' + '_'+ graph_name + '.hierarchy', 'w+') 
    for layer in range(nb_layers-1):
        file_list_ohmnet.write('./Save_graphs/' + 'ohmnet'+'_graph_' + 'training' + '_' + graph_name + str(layer+1) +'\n' )
        file_hierarchy_ohmnet.write('brain_NODE' + ' ' +  './Save_graphs/' + 'ohmnet' +'_graph_' + 'training' + '_' + graph_name + str(layer+1) +'\n')
    file_list_ohmnet.close()
    file_hierarchy_ohmnet.close()
    os.replace('ohmnet_graphlist' + 'training' + '_'+ graph_name + '.list', './Save_graphs/' + 'ohmnet_graphlist' + 'training' + '_'+ graph_name + '.list')
    os.replace('ohmnet_hierarchy_' + 'training' + '_'+ graph_name + '.hierarchy', './Save_graphs/'+'ohmnet_hierarchy_' + 'training' + '_'+ graph_name + '.hierarchy')
   
    # Write graph collapsed nut no longer used
    graph_collapsed = fnumba.preprocess('collapsed_original' + '_'+ graph_name, './Save_graphs', ' ', directed = False,  relabel = True,  del_self_loops = True) #directed, relabel, del_self_loops
    collapsed_traintest_split = EvalSplit()    
    collapsed_traintest_split.compute_splits(graph_collapsed, split_alg=split_alg, train_frac=train_frac, owa=False)
    nee_graph_collapsed = LPEvaluator(collapsed_traintest_split, dim=EMBED_DIMENSION)
    Collapsed_split = collapsed_traintest_split.TG
    nx.write_edgelist(Collapsed_split, 'graphNode2vec' +graph_name+'.txt', data=False)
  
    ########################################################################################
    # EVALUATE BASELINE
    ########################################################################################
    results_embeddings_methods = dict()
    methods = list()
      
    for layer in range(nb_layers-1):  
        scoresheet_baseline = Scoresheet(tr_te='test')    
        fnumba.eval_baselines(nee[layer], False, scoresheet_baseline)
        tmp = scoresheet_baseline.get_pandas_df()        
        for i in range (len(tmp)):
            results_embeddings_methods[tmp.index[i]+ '_'+str(layer)]=np.float(tmp.values[i])       
            methods.append(tmp.index[i]+ '_'+str(layer))
            
     
    methods = ['common_neighbours', 'jaccard_coefficient', 'adamic_adar_index',
               'preferential_attachment']  
    
    for method in methods:   
        graph_ee_train = []
        graph_ee_test = []
        df_sim_train = []
        df_sim_test = []
        df_concat=[]

        for layer in range(nb_layers-1):
        # Evaluate baseline methods
            if method == 'common_neighbours':
                graph_ee_train.append(similarity.common_neighbours(Gsplit[layer], traintestsplit[layer].train_edges))
                graph_ee_test.append(similarity.common_neighbours(Gsplit[layer], traintestsplit[layer].test_edges))
                
            elif method == 'jaccard_coefficient':
                graph_ee_train.append(similarity.jaccard_coefficient(Gsplit[layer], traintestsplit[layer].train_edges))
                graph_ee_test.append(similarity.jaccard_coefficient(Gsplit[layer], traintestsplit[layer].test_edges))
    
            elif method == 'adamic_adar_index':
                graph_ee_train.append(similarity.adamic_adar_index(Gsplit[layer], traintestsplit[layer].train_edges))
                graph_ee_test.append(similarity.adamic_adar_index(Gsplit[layer], traintestsplit[layer].test_edges))
    
            elif method == 'preferential_attachment':
                graph_ee_train.append(similarity.preferential_attachment(Gsplit[layer], traintestsplit[layer].train_edges))
                graph_ee_test.append(similarity.preferential_attachment(Gsplit[layer], traintestsplit[layer].test_edges))
                
            
            df_train = pd.DataFrame(traintestsplit[layer].train_edges)
            df_train.insert(2, 'train', graph_ee_train[layer])
            df_train[0] = df_train[0].apply(str)
            df_train[1] = df_train[1].apply(str)
            df_train = df_train.set_index([0,1])
            df_sim_train.append(df_train)
            
            df_test = pd.DataFrame(traintestsplit[layer].test_edges)
            df_test.insert(2, 'test',graph_ee_test[layer])
            df_test[0] = df_test[0].apply(str)
            df_test[1] = df_test[1].apply(str)
            df_test = df_test.set_index([0,1])
            df_sim_test.append(df_test)
        
        df_concat_train = pd.concat(df_sim_train)
        by_row_index_train = df_concat_train.groupby(df_concat_train.index)
        df_means_train = by_row_index_train.mean()
        
        df_concat_test = pd.concat(df_sim_test)
        by_row_index_test = df_concat_test.groupby(df_concat_test.index)
        df_means_test = by_row_index_test.mean()
                        
        
        for layer in range(nb_layers-1):    
            train_edges_layer = traintestsplit[layer].train_edges
            train_edges_layer = tuple((str(x[0]), str(x[1])) for x in train_edges_layer)
            test_edges_layer = traintestsplit[layer].test_edges
            test_edges_layer = tuple((str(x[0]), str(x[1])) for x in test_edges_layer)
            
            df_means_train_layer = df_means_train[df_means_train.index.isin(train_edges_layer)]
            df_means_test_layer = df_means_test[df_means_test.index.isin(test_edges_layer)]
            
            train_pred, test_pred = nee[layer].compute_pred(data_split=traintestsplit[layer], tr_edge_embeds=df_means_train_layer,
                                                          te_edge_embeds=df_means_test_layer)
            
            results = nee[layer].compute_results(data_split=traintestsplit[layer], method_name=method,
                                               train_pred=train_pred, test_pred=test_pred)
               
            results_embeddings_methods[method+ '_'+str(layer)]=results.get_all()[1][4]
        
    
    ########################################################################################
    # EVALUATE Node2vec, Deepwalk and others
    ########################################################################################
    edge_emb = ['hadamard', 'weighted_l1', 'weighted_l2', 'average', 'cosine']
    G=Gr()

    X_node2vec=[]
    for layer in range(nb_layers-1):   

        G=Gr()
        G.read_edgelist('./Save_graphs/' + 'ohmnet'+'_graph_' + 'training' + '_' + graph_name + str(layer+1), directed=False)
        n2vec_model = Node2vec(G, 10, 10, EMBED_DIMENSION, p=1, q=2, workers=48, window = 10)
        n2vec_model.save_embeddings('embeddings' + '_' + 'Node2vec' + '_' + graph_name)
        os.replace('./' + 'embeddings' + '_' + 'Node2vec' + '_' + graph_name , './Save_embeddings/' + 'embeddings' + '_' + 'Node2vec' + '_' + graph_name)
        X_node2vec.append(pd.DataFrame.from_dict(n2vec_model.vectors))
        
    df_concat = pd.concat(X_node2vec)
    by_row_index = df_concat.groupby(df_concat.index)
    df_means = by_row_index.mean()
    X = df_means.to_dict('list')   
    X = {k: np.array(v) for k, v in X.items()}
        
    for layer in range(nb_layers-1):    
        for i in range (len(edge_emb)):
            tmp_result_node2vec = nee[layer].evaluate_ne(data_split=nee[layer].traintest_split, X=X, method="Node2vec_Layer", edge_embed_method=edge_emb[i], 
                                   label_binarizer=LogisticRegressionCV(solver=solver,  max_iter=max_iter, n_jobs=48, penalty='l2'))
            results_embeddings_methods[tmp_result_node2vec.method+'_'+str(layer) + str(edge_emb[i])] = tmp_result_node2vec.get_all()[1][4]

    # DEEPWALK  
    X_deepwalk = []
    for layer in range(nb_layers-1):   

        G=Gr()
        G.read_edgelist('./Save_graphs/' + 'ohmnet'+'_graph_' + 'training' + '_' + graph_name + str(layer+1), directed=False)
           
        deepwalk_model = Node2vec(G, 10, 10 , EMBED_DIMENSION, p=1, q=1, workers=48, window = 10)
        deepwalk_model.save_embeddings('embeddings'+ '_' + 'Deepwalk' + '_' + graph_name)
        os.replace('./' + 'embeddings' + '_' + 'Deepwalk' + '_' + graph_name , './Save_embeddings/' + 'embeddings' + '_' + 'Deepwalk' + '_' + graph_name)        
        X_deepwalk.append(pd.DataFrame.from_dict(deepwalk_model.vectors))

    df_concat = pd.concat(X_deepwalk)
    by_row_index = df_concat.groupby(df_concat.index)
    df_means = by_row_index.mean()
    X = df_means.to_dict('list')   
    X = {k: np.array(v) for k, v in X.items()}
    
    for layer in range(nb_layers-1):    
        for i in range (len(edge_emb)):
            tmp_result_deepwalk = nee[layer].evaluate_ne(data_split=nee[layer].traintest_split, X=X, method="Deepwalk_Layer", edge_embed_method=edge_emb[i], 
                                   label_binarizer=LogisticRegressionCV(solver=solver,  max_iter=max_iter, n_jobs=48, penalty='l2'))
            results_embeddings_methods[tmp_result_deepwalk.method+'_'+str(layer) + str(edge_emb[i])] = tmp_result_deepwalk.get_all()[1][4]

    #   LINE  
    X_LINE = []
    for layer in range(nb_layers-1):   

        G=Gr()
        G.read_edgelist('./Save_graphs/' + 'ohmnet'+'_graph_' + 'training' + '_' + graph_name + str(layer+1), directed=False)
           
        line_model = LINE(G, rep_size=EMBED_DIMENSION, batch_size=max_iter, negative_ratio=5, order=3)
        line_model.save_embeddings('embeddings'+ '_' + 'LINE' + '_' + graph_name)
        os.replace('./' + 'embeddings' + '_' + 'LINE' + '_' + graph_name , './Save_embeddings/' + 'embeddings' + '_' + 'LINE' + '_' + graph_name)        
        X_LINE.append(pd.DataFrame.from_dict(line_model.vectors))  
        del line_model

    df_concat = pd.concat(X_LINE)
    by_row_index = df_concat.groupby(df_concat.index)
    df_means = by_row_index.mean()
    X = df_means.to_dict('list')   
    X = {k: np.array(v) for k, v in X.items()}
    
    for layer in range(nb_layers-1):    
        for i in range (len(edge_emb)):
            tmp_result_LINE = nee[layer].evaluate_ne(data_split=nee[layer].traintest_split, X=X, method="LINE_Layer", edge_embed_method=edge_emb[i], 
                                   label_binarizer=LogisticRegressionCV(solver=solver,  max_iter=max_iter, n_jobs=48, penalty='l2'))
            results_embeddings_methods[tmp_result_LINE.method+'_'+str(layer) + str(edge_emb[i])] = tmp_result_LINE.get_all()[1][4]
    
    
    ########################################################################################
    # EVALUATE multiplex network embedding OMHNET
    ########################################################################################
    
    from ohmnet import ohmnet
    on = ohmnet.OhmNet( net_input= './Save_graphs/'+'ohmnet_graphlist' + 'training' + '_'+ graph_name + '.list', weighted=False, directed=False,     
                           hierarchy_input='./Save_graphs/'+ 'ohmnet_hierarchy_' + 'training' + '_'+ graph_name + '.hierarchy', p=1, q=1, num_walks=10,
                           walk_length=10, dimension=EMBED_DIMENSION,
                           window_size=10, n_workers=48, n_iter=5,
                           out_dir='./Save_embeddings/'+'ohmnet'+graph_name+'/')
    on.embed_multilayer()
    
    from itertools import islice
    with open('./Save_embeddings/'+'ohmnet'+graph_name+'/'+'internal_vectors.emb', 'r') as document:
        ohmnet_emb = {}
        for line in islice(document, 1, None):
            if line.strip():  # non-empty line?
                key, value = line.split(None, 1)  # None means 'all whitespace', the default
                key = [np.int(s) for s in key.split("__") if s.isdigit()]
                ohmnet_emb[key.pop()] = np.array( value.split(), dtype=np.float64)            

    X = ohmnet_emb
    X = {str(key):X[key] for key in X}

    for layer in range(nb_layers-1):   
        for i in range (len(edge_emb)):
            tmp_result_ohmnet = nee[layer].evaluate_ne(data_split=nee[layer].traintest_split, X=X, method="Ohmnet", edge_embed_method=edge_emb[i],  label_binarizer=lp_model )
            results_embeddings_methods[tmp_result_ohmnet.method+'_'+str(layer) + str(edge_emb[i])] = tmp_result_ohmnet.get_all()[1][4]
 
    ########################################################################################
    # EVALUATE multiplex network embedding MNE
    ########################################################################################
    
    import MNE as mne
    file_name ='./Save_graphs/'+ 'multiverse_graph_' + 'training' + '_'+ graph_name+'.txt'

    edge_data_by_type_mne, all_edges_mne, all_nodes_mne = load_network_data(file_name)
    mne_emb = mne.train_model(edge_data_by_type_mne)
    X = dict(zip(range(mne_emb['base'].shape[0]), mne_emb['base']))
    kv = list(X.items())
    X.clear()
    for k, v in kv:
        X[mne_emb['index2word'][k]] = v
        
    for layer in range(nb_layers-1):   
        for i in range (len(edge_emb)):
            tmp_result_mne = nee[layer].evaluate_ne(data_split=nee[layer].traintest_split, X=X, method="MNE", edge_embed_method=edge_emb[i], label_binarizer=lp_model )
            results_embeddings_methods[tmp_result_mne.method+'_'+str(layer) + str(edge_emb[i])] = tmp_result_mne.get_all()[1][4]    
    
    
    ###################################################################################"
    # Multi-node2vec
    ###################################################################################"
    
    for layer in range(nb_layers-1):   
        G=Gr()
        G.read_edgelist('./Save_graphs/' + 'ohmnet'+'_graph_' + 'training' + '_' + graph_name + str(layer+1), directed=False)
        A = nx.to_pandas_adjacency(G.G, order=G.G.nodes)

        
        if not os.path.exists('./Save_graphs/Multinode2vec_graphs/'+graph_name+'/'):
            os.mkdir('./Save_graphs/Multinode2vec_graphs/'+graph_name+'/')
        A.to_csv('./Save_graphs/Multinode2vec_graphs/'+graph_name+'/'+'net_multinode2vec'+ graph_name + str(layer)+'.csv')
    
    proc = subprocess.Popen(['python3',  './multi-node2vec/multi_node2vec.py', \
              '--dir', './Save_graphs/Multinode2vec_graphs/'+graph_name+'/', '--output', \
              './embedmultinode2vec/'+graph_name+'/','--d', str(EMBED_DIMENSION), '--walk_length', '100', '--w2v_workers', '40'])

    proc.wait()   
    pid = proc.pid 
    proc.kill()    
    
    multinode2vec_emb = pd.read_csv('./embedmultinode2vec_NR/'+ graph_name +'/r0.25/mltn2v_results.csv', header=None)
    multinode2vec_emb.index = multinode2vec_emb[0]
    multinode2vec_emb = multinode2vec_emb.drop(columns=0, axis=1)
    X = multinode2vec_emb.T.to_dict('list')    
    X = {k: np.array(v) for k, v in X.items()}  
    X = {str(key):X[key] for key in X}


    for layer in range(nb_layers-1):   
        for i in range (len(edge_emb)):
            tmp_result_Multinode2vec = nee[layer].evaluate_ne(data_split=nee[layer].traintest_split, X=X, method="MultiNode2vec", edge_embed_method=edge_emb[i], label_binarizer=lp_model )
            results_embeddings_methods[tmp_result_Multinode2vec.method+'_'+str(layer) + str(edge_emb[i])] = tmp_result_Multinode2vec.get_all()[1][4]    
    
    
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
#    os.system('Rscript  ./Multiverse-ForEach/GenerateSimMatrix.R \
#              -n ../Save_graphs/'+'multiverse_graph_' + 'training' + '_'+ graph_name+'.txt'+' -o \
#              ../ResultsRWR/MatrixSimilarityMultiplex'+graph_name+' -c 10')
#    
    os.system('module unload R/3.4.0')

    print('bye')
    r_DistancematrixPPI = r_readRDS('./ResultsRWR/MatrixSimilarityMultiplex'+graph_name +'.rds') 

    import gc
    gc.collect()

        ########################################################################
        # Processing of the network
        ########################################################################
    reverse_data_DistancematrixPPI, list_neighbours, nodes, data_DistancematrixPPI, nodes_incomponent, nodesstr \
     = fnumba.netpreprocess_optimized(r_DistancematrixPPI, graph_path, KL, CLOSEST_NODES)
     
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
    tmp_Node2vec_Result_hada = 0
    tmp_Node2vec_Result_wl1 = 0
    tmp_Node2vec_Result_wL2 = 0
    tmp_Node2vec_Result_avg = 0
    tmp_Node2vec_Result_cos = 0
    tmp_Deepwalk_Result_hada = 0
    tmp_Deepwalk_Result_wl1 = 0
    tmp_Deepwalk_Result_wl2 = 0
    tmp_Deepwalk_Result_avg = 0
    tmp_Deepwalk_Result_cos = 0
    tmp_LINE_Result_hada = 0
    tmp_LINE_Result_wl1 = 0
    tmp_LINE_Result_wl2 = 0
    tmp_LINE_Result_avg = 0
    tmp_LINE_Result_cos = 0
    tmp_Ohmnet_Result_hada = 0
    tmp_Ohmnet_Result_wl1 = 0
    tmp_Ohmnet_Result_wl2 = 0
    tmp_Ohmnet_Result_avg = 0
    tmp_Ohmnet_Result_cos = 0
    tmp_MNE_Result_hada = 0
    tmp_MNE_Result_wl1 = 0
    tmp_MNE_Result_wl2 = 0
    tmp_MNE_Result_avg = 0
    tmp_MNE_Result_cos = 0
    tmp_MultiNode2vec_Result_hada = 0
    tmp_MultiNode2vec_Result_wl1 = 0
    tmp_MultiNode2vec_Result_wl2 = 0
    tmp_MultiNode2vec_Result_avg = 0
    tmp_MultiNode2vec_Result_cos = 0
    tmp_common_Result = 0
    tmp_jaccard_Result = 0
    tmp_adamic_Result = 0
    tmp_prefattach_Result=0
    
    for layer in range(nb_layers-1):
        tmp_Multiverse_Result_hada += results_embeddings_methods['Multiverse'+'_'+str(layer) + str(edge_emb[0])]
        tmp_Multiverse_Result_wl1 += results_embeddings_methods['Multiverse'+'_'+str(layer) + str(edge_emb[1])]
        tmp_Multiverse_Result_wL2 += results_embeddings_methods['Multiverse'+'_'+str(layer) + str(edge_emb[2])]
        tmp_Multiverse_Result_avg += results_embeddings_methods['Multiverse'+'_'+str(layer) + str(edge_emb[3])]
        tmp_Multiverse_Result_cos += results_embeddings_methods['Multiverse'+'_'+str(layer) + str(edge_emb[4])]

        tmp_Ohmnet_Result_hada += results_embeddings_methods['Ohmnet'+'_'+str(layer) + str(edge_emb[0])]
        tmp_Ohmnet_Result_wl1 += results_embeddings_methods['Ohmnet'+'_'+str(layer) + str(edge_emb[1])]
        tmp_Ohmnet_Result_wl2 += results_embeddings_methods['Ohmnet'+'_'+str(layer) + str(edge_emb[2])]
        tmp_Ohmnet_Result_avg += results_embeddings_methods['Ohmnet'+'_'+str(layer) + str(edge_emb[3])]
        tmp_Ohmnet_Result_cos += results_embeddings_methods['Ohmnet'+'_'+str(layer) + str(edge_emb[4])]

        tmp_MultiNode2vec_Result_hada += results_embeddings_methods['MultiNode2vec'+'_'+str(layer) + str(edge_emb[0])]
        tmp_MultiNode2vec_Result_wl1 += results_embeddings_methods['MultiNode2vec'+'_'+str(layer) + str(edge_emb[1])]
        tmp_MultiNode2vec_Result_wl2 += results_embeddings_methods['MultiNode2vec'+'_'+str(layer) + str(edge_emb[2])]
        tmp_MultiNode2vec_Result_avg += results_embeddings_methods['MultiNode2vec'+'_'+str(layer) + str(edge_emb[3])]
        tmp_MultiNode2vec_Result_cos += results_embeddings_methods['MultiNode2vec'+'_'+str(layer) + str(edge_emb[4])]
        
        tmp_MNE_Result_hada += results_embeddings_methods['MNE'+'_'+str(layer) + str(edge_emb[0])]
        tmp_MNE_Result_wl1 += results_embeddings_methods['MNE'+'_'+str(layer) + str(edge_emb[1])]
        tmp_MNE_Result_wl2 += results_embeddings_methods['MNE'+'_'+str(layer) + str(edge_emb[2])]
        tmp_MNE_Result_avg += results_embeddings_methods['MNE'+'_'+str(layer) + str(edge_emb[3])]
        tmp_MNE_Result_cos += results_embeddings_methods['MNE'+'_'+str(layer) + str(edge_emb[4])]
        
        tmp_common_Result += results_embeddings_methods['common_neighbours'+'_'+str(layer)]
        tmp_jaccard_Result += results_embeddings_methods['jaccard_coefficient'+'_'+str(layer)]
        tmp_adamic_Result += results_embeddings_methods['adamic_adar_index'+'_'+str(layer)] 
        tmp_prefattach_Result += results_embeddings_methods['preferential_attachment'+'_'+str(layer)]
      
        tmp_Node2vec_Result_hada += results_embeddings_methods['Node2vec_Layer'+'_'+str(layer) + str(edge_emb[0])]
        tmp_Node2vec_Result_wl1 += results_embeddings_methods['Node2vec_Layer'+'_' +str(layer)+ str(edge_emb[1])]
        tmp_Node2vec_Result_wL2 += results_embeddings_methods['Node2vec_Layer'+'_'+str(layer)+ str(edge_emb[2])]
        tmp_Node2vec_Result_avg += results_embeddings_methods['Node2vec_Layer'+'_'+str(layer)+ str(edge_emb[3])]
        tmp_Node2vec_Result_cos += results_embeddings_methods['Node2vec_Layer'+'_'+str(layer)+ str(edge_emb[4])]
    
        tmp_Deepwalk_Result_hada += results_embeddings_methods['Deepwalk_Layer'+'_'+str(layer)+ str(edge_emb[0])]
        tmp_Deepwalk_Result_wl1 += results_embeddings_methods['Deepwalk_Layer'+'_'+str(layer)+ str(edge_emb[1])]
        tmp_Deepwalk_Result_wl2 += results_embeddings_methods['Deepwalk_Layer'+'_'+str(layer)+ str(edge_emb[2])]
        tmp_Deepwalk_Result_avg += results_embeddings_methods['Deepwalk_Layer'+'_'+str(layer)+ str(edge_emb[3])]
        tmp_Deepwalk_Result_cos += results_embeddings_methods['Deepwalk_Layer'+'_'+str(layer)+ str(edge_emb[4])]
    
        tmp_LINE_Result_hada += results_embeddings_methods['LINE_Layer'+'_'+str(layer)+ str(edge_emb[0])]
        tmp_LINE_Result_wl1 += results_embeddings_methods['LINE_Layer'+'_' +str(layer)+ str(edge_emb[1])]
        tmp_LINE_Result_wl2 += results_embeddings_methods['LINE_Layer'+'_'+str(layer)+ str(edge_emb[2])]
        tmp_LINE_Result_avg += results_embeddings_methods['LINE_Layer'+'_'+str(layer) + str(edge_emb[3])]
        tmp_LINE_Result_cos += results_embeddings_methods['LINE_Layer'+'_'+str(layer) + str(edge_emb[4])]

    results_embeddings_methods['Multiverse_av_hadamard'] = tmp_Multiverse_Result_hada/(nb_layers-1)
    results_embeddings_methods['Multiverse_av_weighted_l1'] = tmp_Multiverse_Result_wl1/(nb_layers-1)
    results_embeddings_methods['Multiverse_av_weighted_l2'] = tmp_Multiverse_Result_wL2/(nb_layers-1)
    results_embeddings_methods['Multiverse_av_average'] = tmp_Multiverse_Result_avg/(nb_layers-1)
    results_embeddings_methods['Multiverse_av_cosine'] = tmp_Multiverse_Result_cos/(nb_layers-1)
   
    results_embeddings_methods['Node2vec_av_hadamard_Layer'] = tmp_Node2vec_Result_hada/(nb_layers-1)
    results_embeddings_methods['Node2vec_av_weighted_l1_Layer'] = tmp_Node2vec_Result_wl1/(nb_layers-1)
    results_embeddings_methods['Node2vec_av_weighted_l2_Layer'] = tmp_Node2vec_Result_wL2/(nb_layers-1)
    results_embeddings_methods['Node2vec_av_average_Layer'] = tmp_Node2vec_Result_avg/(nb_layers-1)
    results_embeddings_methods['Node2vec_av_cosine_Layer'] = tmp_Node2vec_Result_cos/(nb_layers-1)

    results_embeddings_methods['Deepwalk_av_hadamard_Layer'] = tmp_Deepwalk_Result_hada/(nb_layers-1)
    results_embeddings_methods['Deepwalk_av_weighted_l1_Layer'] = tmp_Deepwalk_Result_wl1/(nb_layers-1)
    results_embeddings_methods['Deepwalk_av_weighted_l2_Layer'] = tmp_Deepwalk_Result_wl2/(nb_layers-1)
    results_embeddings_methods['Deepwalk_av_average_Layer'] = tmp_Deepwalk_Result_avg/(nb_layers-1)
    results_embeddings_methods['Deepwalk_av_cosine_Layer'] = tmp_Deepwalk_Result_cos/(nb_layers-1)

    results_embeddings_methods['LINE_av_hadamard_Layer'] = tmp_LINE_Result_hada/(nb_layers-1)
    results_embeddings_methods['LINE_av_weighted_l1_Layer'] = tmp_LINE_Result_wl1/(nb_layers-1)
    results_embeddings_methods['LINE_av_weighted_l2_Layer'] = tmp_LINE_Result_wl2/(nb_layers-1)
    results_embeddings_methods['LINE_av_average_Layer'] = tmp_LINE_Result_avg/(nb_layers-1)
    results_embeddings_methods['LINE_av_cosine_Layer'] = tmp_LINE_Result_cos/(nb_layers-1)
    
    results_embeddings_methods['Ohmnet_av_hadamard'] = tmp_Ohmnet_Result_hada/(nb_layers-1)
    results_embeddings_methods['Ohmnet_av_weighted_l1'] = tmp_Ohmnet_Result_wl1/(nb_layers-1)
    results_embeddings_methods['Ohmnet_av_weighted_l2'] = tmp_Ohmnet_Result_wl2/(nb_layers-1)
    results_embeddings_methods['Ohmnet_av_average'] = tmp_Ohmnet_Result_avg/(nb_layers-1)
    results_embeddings_methods['Ohmnet_av_cosine'] = tmp_Ohmnet_Result_cos/(nb_layers-1)
 
    results_embeddings_methods['MultiNode2vec_av_hadamard'] = tmp_MultiNode2vec_Result_hada/(nb_layers-1)
    results_embeddings_methods['MultiNode2vec_av_weighted_l1'] = tmp_MultiNode2vec_Result_wl1/(nb_layers-1)
    results_embeddings_methods['MultiNode2vec_av_weighted_l2'] = tmp_MultiNode2vec_Result_wl2/(nb_layers-1)
    results_embeddings_methods['MultiNode2vec_av_average'] = tmp_MultiNode2vec_Result_avg/(nb_layers-1)
    results_embeddings_methods['MultiNode2vec_av_cosine'] = tmp_MultiNode2vec_Result_cos/(nb_layers-1)
    
    results_embeddings_methods['MNE_av_hadamard'] = tmp_MNE_Result_hada/(nb_layers-1)
    results_embeddings_methods['MNE_av_weighted_l1'] = tmp_MNE_Result_wl1/(nb_layers-1)
    results_embeddings_methods['MNE_av_weighted_l2'] = tmp_MNE_Result_wl2/(nb_layers-1)
    results_embeddings_methods['MNE_av_average'] = tmp_MNE_Result_avg/(nb_layers-1)
    results_embeddings_methods['MNE_av_cosine'] = tmp_MNE_Result_cos/(nb_layers-1)

    results_embeddings_methods['common_neighbours'] =  tmp_common_Result/(nb_layers-1)
    results_embeddings_methods['jaccard_coefficient'] = tmp_jaccard_Result/(nb_layers-1)
    results_embeddings_methods['adamic_adar_index'] =  tmp_adamic_Result/(nb_layers-1)
    results_embeddings_methods['preferential_attachment'] = tmp_prefattach_Result/(nb_layers-1)
  
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
       
       print('', file=overall_result)
       print('', file=overall_result)
       print('', file=overall_result)

       print('Overall MELL AUC:', results_embeddings_methods['MELL'], file=overall_result)
       
       print('', file=overall_result)
       print('', file=overall_result)
       print('', file=overall_result)
       
       print('Overall Node2vec AUC hadamard on multiplex:', results_embeddings_methods['Node2vec_av_hadamard_Layer'], file=overall_result)
       print('Overall Node2vec AUC weighted_l1 on multiplex:', results_embeddings_methods['Node2vec_av_weighted_l1_Layer'], file=overall_result)
       print('Overall Node2vec AUC weighted_l2 on multiplex:', results_embeddings_methods['Node2vec_av_weighted_l2_Layer'], file=overall_result)
       print('Overall Node2vec AUC average on multiplex:', results_embeddings_methods['Node2vec_av_average_Layer'], file=overall_result)
       print('Overall Node2vec AUC cosine on multiplex:', results_embeddings_methods['Node2vec_av_cosine_Layer'], file=overall_result)
      

       print('', file=overall_result)
       print('', file=overall_result)
       print('', file=overall_result)

       print('Overall Deepwalk AUC hadamard on multiplex:', results_embeddings_methods['Deepwalk_av_hadamard_Layer'], file=overall_result)
       print('Overall Deepwalk AUC weighted_l1 on multiplex:', results_embeddings_methods['Deepwalk_av_weighted_l1_Layer'], file=overall_result)
       print('Overall Deepwalk AUC weighted_l2 on multiplex:', results_embeddings_methods['Deepwalk_av_weighted_l2_Layer'], file=overall_result)
       print('Overall Deepwalk AUC average on multiplex:', results_embeddings_methods['Deepwalk_av_average_Layer'], file=overall_result)
       print('Overall Deepwalk AUC cosine on multiplex:', results_embeddings_methods['Deepwalk_av_cosine_Layer'], file=overall_result)
      
       print('', file=overall_result)
       print('', file=overall_result)
       print('', file=overall_result)

       print('Overall LINE AUC hadamard on multiplex:', results_embeddings_methods['LINE_av_hadamard_Layer'], file=overall_result)
       print('Overall LINE AUC weighted_l1 on multiplex:', results_embeddings_methods['LINE_av_weighted_l1_Layer'], file=overall_result)
       print('Overall LINE AUC weighted_l2 on multiplex:', results_embeddings_methods['LINE_av_weighted_l2_Layer'], file=overall_result)
       print('Overall LINE AUC average on multiplex:', results_embeddings_methods['LINE_av_average_Layer'], file=overall_result)
       print('Overall LINE AUC cosine on multiplex:', results_embeddings_methods['LINE_av_cosine_Layer'], file=overall_result)
       
       print('', file=overall_result)
       print('', file=overall_result)
       print('', file=overall_result)
       
       print('Overall Ohmnet AUC hadamard:', results_embeddings_methods['Ohmnet_av_hadamard'], file=overall_result)
       print('Overall Ohmnet AUC weighted_l1:', results_embeddings_methods['Ohmnet_av_weighted_l1'], file=overall_result)
       print('Overall Ohmnet AUC weighted_l2:', results_embeddings_methods['Ohmnet_av_weighted_l2'], file=overall_result)
       print('Overall Ohmnet AUC average:', results_embeddings_methods['Ohmnet_av_average'], file=overall_result)
       print('Overall Ohmnet AUC cosine:', results_embeddings_methods['Ohmnet_av_cosine'], file=overall_result)

       print('', file=overall_result)
       print('', file=overall_result)
       print('', file=overall_result)
       
       
       print('Overall MultiNode2vec AUC hadamard:', results_embeddings_methods['MultiNode2vec_av_hadamard'], file=overall_result)
       print('Overall MultiNode2vec AUC weighted_l1:', results_embeddings_methods['MultiNode2vec_av_weighted_l1'], file=overall_result)
       print('Overall MultiNode2vec AUC weighted_l2:', results_embeddings_methods['MultiNode2vec_av_weighted_l2'], file=overall_result)
       print('Overall MultiNode2vec AUC average:', results_embeddings_methods['MultiNode2vec_av_average'], file=overall_result)
       print('Overall MultiNode2vec AUC cosine:', results_embeddings_methods['MultiNode2vec_av_cosine'], file=overall_result)

       print('', file=overall_result)
       print('', file=overall_result)
       print('', file=overall_result)
       print('Overall MNE AUC hadamard:', results_embeddings_methods['MNE_av_hadamard'], file=overall_result)
       print('Overall MNE AUC weighted_l1:', results_embeddings_methods['MNE_av_weighted_l1'], file=overall_result)
       print('Overall MNE AUC weighted_l2:', results_embeddings_methods['MNE_av_weighted_l2'], file=overall_result)
       print('Overall MNE AUC average:', results_embeddings_methods['MNE_av_average'], file=overall_result)
       print('Overall MNE AUC cosine:', results_embeddings_methods['MNE_av_cosine'], file=overall_result)

       print('', file=overall_result)
       print('', file=overall_result)
       print('', file=overall_result)
             
       print('Overall Common Neigh AUC on aggregated:', results_embeddings_methods['common_neighbours'], file=overall_result)
       print('Overall AA AUC on aggregated:', results_embeddings_methods['adamic_adar_index'], file=overall_result)
       print('Overall Jaccard AUC on aggregated:', results_embeddings_methods['jaccard_coefficient'], file=overall_result)        
       print('Overall Preferential attachment AUC on aggregated:', results_embeddings_methods['preferential_attachment'], file=overall_result)   
       
       
       
    overall_result.close() 
    os.replace(Result_file, './Save_results/'+ Result_file)
    
    print('End')

if __name__ == "__main__":
    main()
