# MultiVERSE
Embedding of Monoplex, Multiplex, Heterogeneous, Multiplex-Heterogeneous and full Multiplex-Heterogeneous Networks.

You can find in this repository the necessary files to use MultiVERSE for multiplex end multiplex-heterogeneous network embedding.
In order to use MultiVERSE, you need the networks to be in extended edgelist format:

            edge_type source target weight
              r1         n1    n2    1
              r2         n2    n3    1        

## Requirements

Python 3:
* rpy2
* gensim (fast_version enabled)
* networkx=2.2
* numba=0.50.1
* scikit-learn=0.21.3 (for evaluation)
* pandas


R>=3.6.1:
* devtools
* igraph
* mclust
* kernlab
* R.matlab
* bc3net
* optparse
* tidyverse


A good way to set up the appropriate environement is to create one with anaconda:

'conda create --name LP rpy2 gensim r numba spyder networkx=2.2 pandas  scikit-learn=0.21.3 joblib matplotlib'

## Folders

* Dataset:
You will find here the datasets of multiplex and multiplex-heterogeneous networks.
* evalne:
This is the folder ot the evalne toolbox used for evaluation of the embeddings (<https://evalne.readthedocs.io/en/latest/index.html>) with the add of the cosine embedding operator.
* Generated_graphs:
The different types of graphs generated by MultiVERSE are stored in this folder.
* RWR: 
You will find here the R files of RWR-M(H) (<https://www.ncbi.nlm.nih.gov/pubmed/30020411>)
* ResultsMultiVERSE:
MultiVERSE will store its outputs: embeddings (a dictionary with index as keys and embeddings of the corresponding node as value) and results of evaluation.
* ResultsRWR:
The matrix of similarity generated by RWR-M(H) will be stored in this folder.


## MultiVERSE on multiplex network:

**MultiVERSE_M.py**

This program allows to apply MultiVERSE on a multiplex network.

The usage is the following:

`python3 MultiVERSE_M.py [options]`

          Options:
                 -k NUMERIC
                   Value of the position of the networks in the list 'Test_Networks'

To use the example, you can write in a terminal the following command:

`python3 MultiVERSE_M.py -m ./Dataset/Multiplex/CKM-Physicians-Innovation_multiplex.edges`

The output of this command is the embedding 'embedding_M.npy' in the directory ResultsMultiVERSE. The embedding is a dictionary with the index as key and the corresponding embedding as value.

**Eval_MultiVERSE_M.py**

This program allows to apply link prediction as evaluation of the multiplex embedding.

The usage is the following:

`python3 Eval_MultiVERSE_M.py [options] `

          Options:

          Options:
                 -m CHARACTER
                   Path to the multiplex

To use the example, you can write in a terminal the following command:

`python3 Eval_MultiVERSE_M.py -m ./Dataset/Multiplex/CKM-Physicians-Innovation_multiplex.edges`
			
The output of this command is the embedding 'embedding_M_eval.npy' in the directory ResultsMultiVERSE. The embedding is a dictionary with the index as key and the corresponding embedding as value. The 
command will also generate the results of the evaluation in a .txt file 'Result_Linkpred_Multiplex_Test_Eval.txt'. 

## MultiVERSE on multiplex-heterogeneous network:

**MultiVERSE_MH.py**

This program allows to apply MultiVERSE on a multiplex-heterogeneous network.

The usage is the following:

`python3 MultiVERSE_MH.py [options] `

          Options:

      		 -n CHARACTER
          Path to the first multiplex network to be used as Input.
          It should be a space separated four column file containing
          the fields: edge_type, source, target and weight.
            edge_type source target weight
             r1        n1    n2    1
             r2        n2    n3    1

       	-m CHARACTER
          Path to the second multiplex network to be used as Input.
          It should be a space separated four column file containing
          the fields: edge_type, source, target and weight.
            edge_type source target weight
             r1        n1    n2    1
             r2        n2    n3    1

       	-b CHARACTER
          Path to the bipartite network to be used as Input.
          It should be a space separated four column file containing
          the fields: edge_type, source, target and weight. Source Nodes
          should be the ones from the first multiplex and target nodes
          from the second.
            edge_type source target weight
             r1        n1    n2    1
             r2        n2    n3    1

To use the example, you can write in a terminal the following command:

`python3 MultiVERSE_MH.py -m ./Dataset/Multiplex_Het/Multiplex_1.txt 
			  -n ./Dataset/Multiplex_Het/Multiplex_2.txt 
			  -b ./Dataset/Multiplex_Het/bipartite.txt`
			  
If you don't have enough memory, you can use the toy example:

`python3 MultiVERSE_MH.py -n ./Dataset/Multiplex_Het/M1_toy.txt -m ./Dataset/Multiplex_Het/M2_toy.txt -b ./Dataset/Multiplex_Het/bipartite_toy.txt`
			  
The output of this command is the embedding 'embedding_MH.npy' in the directory ResultsMultiVERSE. The embedding is a dictionary with the index as key and the corresponding embedding as value.

**Eval_MultiVERSE_MH.py**

This program allows to apply link prediction as evaluation of the multiplex-heterogeneous embedding.

The usage is the following:

`python3 Eval_MultiVERSE_MH.py [options] `

          Options:

      		 -n CHARACTER
          Path to the first multiplex network to be used as Input.
          It should be a space separated four column file containing
          the fields: edge_type, source, target and weight.
            edge_type source target weight
             r1        n1    n2    1
             r2        n2    n3    1

       	-m CHARACTER
          Path to the second multiplex network to be used as Input.
          It should be a space separated four column file containing
          the fields: edge_type, source, target and weight.
            edge_type source target weight
             r1        n1    n2    1
             r2        n2    n3    1

       	-b CHARACTER
          Path to the bipartite network to be used as Input.
          It should be a space separated four column file containing
          the fields: edge_type, source, target and weight. Source Nodes
          should be the ones from the first multiplex and target nodes
          from the second.
            edge_type source target weight
             r1        n1    n2    1
             r2        n2    n3    1

To use the example, you can write in a terminal the following command:

`python3 Eval_MultiVERSE_MH.py -m ./Dataset/Multiplex_Het/Multiplex_1.txt 
                               -n ./Dataset/Multiplex_Het/Multiplex_2.txt 
                               -b ./Dataset/Multiplex_Het/bipartite.txt`
                               
If you don't have enough memory, you can use the toy example:

`python3 Eval_MultiVERSE_MH.py -n ./Dataset/Multiplex_Het/M1_toy.txt -m ./Dataset/Multiplex_Het/M2_toy.txt -b ./Dataset/Multiplex_Het/bipartite_toy.txt`
			
The output of this command is the embedding 'embedding_MH_eval.npy' in the directory ResultsMultiVERSE. The embedding is a dictionary with the index as key and the corresponding embedding as value. The command will also generate the results of the evaluation in a .txt file 'Result_LinkpredMultiplexHet_Test_Eval.txt'. 
 
## Usage of the RWR files:

In the RWR folder, you will find:
* *GenerateSimMatrix.R:* Script that computes RWR scores taking as a seed every
individual node of the input network. These scores are used to build a N x N
matrix,where N is the number of nodes of the input network. The goal is to apply
network embedding methods on this matrix. To apply for monoplex and multiplex networks.
For additional information about RWR on multiplex network, we refer to:
<https://www.ncbi.nlm.nih.gov/pubmed/30020411>
* *GenerateSimMatrix_MH.R:* This script is similar to the previous one but it can applied to a heterogeneous, a multiplex-heterogeneous or a full multiplex-heterogeneous network. The RWR scores are used to build a (N+M) x (N+M) matrix, where N and M are the number of nodes of the first and second network respectively. The goal is to apply network embedding methods on this matrix. For additional information about RWR
on heterogeneous and multiplex-heterogeneous network, we refer to: <https://www.ncbi.nlm.nih.gov/pubmed/30020411>
* *EdgelistToMultiplex.R:* Script that takes as input monoplex networks on
edgelist format and transforms then to the multiplex format required as input
in the *GenerateSimMatrix.R* script.
* *Functions_RWRMH.R:* An R file containing the functions to perform RWR on
multiplex networks used by the *GenerateSimMatrix.R* script.
* *Networks:* A folder contaning some input networks in edgelist format and others with the format required as input for the
*GenerateSimMatrix.R* script.


**1.- GenerateSimMatrix.R**

Type on the command line the following command:

`Rscript GenerateSimMatrix.R [options]`

    Options:
       -n CHARACTER, --network=CHARACTER
          Path to the multiplex network to be used as Input. It should be a  
          space separated four column file containing the fields:
          edge_type, source, target and weight:

          edge_type source target weight
             r1         n1    n2    1
             r2         n2    n3    1

       -r NUMERIC, --restart=NUMERIC
         Value of the restart parameter ranging from 0 to 1. [default= 0.7]

       -o CHARACTER, --out=CHARACTER
           Name for the output file (The resulting Similarity Matrix saved as an
           rds file. rds extension will be attached to the name provided here).

       -c INTEGER, --cores=INTEGER
       Number of cores to be used for the Random Walk Calculation. [default= 1]

       -h, --help
           Show this help message and exit

*Example:* We compute the similarity matrix of the multiplex network generated in the example described in section 2 (below). In this case, we used 4 cores:

`Rscript GenerateSimMatrix.R -n ./Networks/ppi_path_RWRMHvignette.txt -o ./Results/SimMatrixTest -c 4 -r 0.5`


**2.- GenerateSimMatrix_MH.R**

Type on the command line the following command:

`Rscript GenerateSimMatrix_MH.R [options]`

    Options:
       -n CHARACTER, --network1=CHARACTER
          Path to the first multiplex network to be used as Input.
          It should be a space separated four column file containing
          the fields: edge_type, source, target and weight.
            edge_type source target weight
             r1        n1    n2    1
             r2        n2    n3    1

       -m CHARACTER, --network2=CHARACTER
          Path to the second multiplex network to be used as Input.
          It should be a space separated four column file containing
          the fields: edge_type, source, target and weight.
            edge_type source target weight
             r1        n1    n2    1
             r2        n2    n3    1

       -b CHARACTER, --bipartite=CHARACTER
          Path to the bipartite network to be used as Input.
          It should be a space separated four column file containing
          the fields: edge_type, source, target and weight. Source Nodes
          should be the ones from the first multiplex and target nodes
          from the second.
            edge_type source target weight
             r1        n1    n2    1
             r2        n2    n3    1

       -r NUMERIC, --restart=NUMERIC
         Value of the restart parameter ranging from 0 to 1. [default= 0.7]

       -o CHARACTER, --out=CHARACTER
         Name for the output file (The resulting Similarity Matrix saved as an
         rds file. rds extension will be attached to the name provided here).

       -c INTEGER, --cores=INTEGER
       Number of cores to be used for the Random Walk Calculation. [default= 1]

       -h, --help
           Show this help message and exit

*Example:* We compute the similarity matrix of the full-multiplex heterogeneous toy network displayed in the figure. The input network files can be found in the networks directory.

`Rscript GenerateSimMatrix_MH.R -n Networks/m1_toy.txt -m Networks/m2_toyMulti.txt -b Networks/bipartite_toy.txt -o example`


**3.- EdgelistToMultiplex.R**

Type on the command line the following command:

`Rscript EdgelistToMultiplex.R [options]`

    Options:
        -n CHARACTER, --networks=CHARACTER
            Path to the monoplex networks to be used as Inputs. They should be
            in edgelist format, with two or three column depending if edges are
            weighted : source node, target node, weight (optional). If more than
            one network is provided, they should be separated by commas:

         source target weight
             n1    n2    1
             n2    n3    1

       -e CHARACTER, --edgeTypes=CHARACTER
           Names for the different layers (Usually the type of edge described by
           the interactions). If more than one network is provided, they should
           be separated by commas.

       -o CHARACTER, --out=CHARACTER
           Name for the output file (The resulting multiplex network)

       -h, --help
           Show this help message and exit

*Example:* Use PPI and Pathway network from the RandomWalkRestartMH package
(<http://bioconductor.org/packages/release/bioc/html/RandomWalkRestartMH.html>)
to generate a 2-layers multiplexnetwork output file

`Rscript EdgelistToMultiplex.R -n ./Networks/PPI_RWR_Vignette.txt,./Networks/Pathway_RWR_Vignette.txt -o ./Networks/ppi_path_RWRMHvignette.txt -e ppi,path`
