#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### GenerateSimMatrix.R: Script that generates the Similarity matrix (output) 
#### of a given multiplex network (input) based on the scores of the RWR
#### (see paper: https://www.ncbi.nlm.nih.gov/pubmed/30020411)
#### been performed on every single node of the network. 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####

#### Usage: 

# Usage: Rscript GenerateSimMatrix.R [options]

#   Options:
#       -n CHARACTER, --network=CHARACTER
#          Path to the multiplex network to be used as Input. It should be a  
#          space separated four column file containing the fields: 
#          edge_type, source, target and weight.
#            edge_type source target weight
#             r1     	  n1    n2    1
#             r2        n2    n3    1

#       -r NUMERIC, --restart=NUMERIC
#         Value of the restart parameter ranging from 0 to 1. [default= 0.7]

#       -o CHARACTER, --out=CHARACTER
#         Name for the output file (The resulting Similarity Matrix saved as an
#         rds file. rds extension will be attached to the name provided here).

#       -c INTEGER, --cores=INTEGER
#       Number of cores to be used for the Random Walk Calculation. [default= 1]

#       -h, --help
#           Show this help message and exit

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### Loading functions and external packages
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
## We clean the workspace
rm(list=ls());cat('\014');if(length(dev.list()>0)){dev.off()}

setwd("./RWR/")

## We load the R file containing the associated RWR functions.
source("Functions_RWRM.R")

## Installation and load of the required R Packages
packages <- c("igraph", "mclust","Matrix","kernlab", "R.matlab","bc3net",
              "optparse","parallel","tidyverse","doParallel")
ipak(packages)

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### Reading and checking the input arguments
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####

## Reading input arguments

#!/usr/bin/env Rscript
message("Reading arguments...")
option_list = list(
  make_option(c("-n", "--network"), type="character", default=NULL, 
              help="Path to the multiplex network to be used as Input. It 
              should be a space separated four column file containing the fields: 
              edge_type, source node, target node, weight", metavar="character"),
  make_option(c("-r", "--restart"), type="double", default=0.7, 
              help="Value of the restart parameter ranging from 0 to 1. 
              [default= %default]", metavar="numeric"),
  make_option(c("-o", "--out"), type="character", default=NULL, 
              help="Name for the output file. SimMatrix will be added before 
              this argument", metavar="character"),
  make_option(c("-c", "--cores"), type="integer", default=1, 
              help="Number of cores to be used for the Random Walk Calculation. 
              [default= %default]", metavar="integer")
); 

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

## Checking if the arguments are valid. Otherwise, stop the execution and
## return an error

if (is.null(opt$network)){
  print_help(opt_parser)
  stop("You need to specify the input network to be used.", call.=FALSE)
}  

InputNetwork <- read.csv(opt$network, header = FALSE, sep=" ")
if (nrow(InputNetwork)==0){
  print_help(opt_parser)
  stop("The input network should have interactions", call.=FALSE)
}

if (ncol(InputNetwork)!=4){
  print_help(opt_parser)
  stop("The input network should has 4 columns. See help for format details", 
       call.=FALSE)
} else {
  colnames(InputNetwork) <- c("EdgeType","source","target","weight")
  Number_Layers <- length(unique(InputNetwork$EdgeType))
  print(paste0("Input multiples network with ", Number_Layers, " Layers"))
  ## We check that the weights are numeric. 
  if (!is.numeric(InputNetwork$weight)){
    print_help(opt_parser)
    stop("The weights in the input network should be numeric", 
         call.=FALSE)
  }
}

if (opt$restart > 1 || opt$restart < 0){
  print_help(opt_parser)
  stop("Restart parameter should range between 0 and 1", call.=FALSE)
}

if (is.null(opt$out)){
  print_help(opt_parser)
  stop("You need to specify a name to be used to generate the output files.", 
       call.=FALSE)
}

MachineCores <- detectCores()

if (opt$cores < 1 || opt$cores > MachineCores){
  print_help(opt_parser)
  stop("The number of cores should be between 1 and the total number of cores of 
       your machine", call.=FALSE)
}


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### Data Transformation and associated calculations to apply RWR
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####

## We transform the input multiplex format to a L-length list of igraphs objects.
## We also scale the weigths of every layer between 1 and the minimun 
## divided by the maximun weight. 

LayersNames <- unique(InputNetwork$EdgeType)
Layers <- lapply(LayersNames, function(x) {
  Original_weight <- InputNetwork$weight[InputNetwork$EdgeType ==x]
  if (!all(Original_weight)){
    b <- 1
    a <- min(Original_weight)/max(Original_weight)
    range01 <- (b-a)*(Original_weight-min(Original_weight))/(max(Original_weight)-min(Original_weight)) + a
    InputNetwork$weight[InputNetwork$EdgeType ==x] <- range01
  }
  
  igraph::simplify(graph_from_data_frame(InputNetwork[InputNetwork$EdgeType == x,2:4],
                  directed = FALSE),edge.attr.comb=mean)
})
names(Layers) <-LayersNames


## In case the network is monoplex, we have to be aware of isolated nodes.
if (Number_Layers == 1){
  message("Dealing with isolated nodes...")
  IsolatedVertex <- V(Layers[[1]])$name[which(degree(Layers[[1]])==0)]
  mynetPrev <- Layers[[1]]
  Layers[[1]] <- delete.vertices(igraph::simplify(Layers[[1]]), 
                                 degree(Layers[[1]])==0) 
}

## We prepate the data to compute RWR (Compute the adjacency matrix of the 
## multiplex network and its normalalisation)

MultiplexObject <- create.multiplex(Layers)
AdjMatrix <- compute.adjacency.matrix(MultiplexObject)
AdjMatrixNorm <- normalize.multiplex.adjacency(AdjMatrix)
Allnodes <- MultiplexObject$Pool_of_Nodes
numberNodes <- length(Allnodes)

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### Apply RWR on every node of the multiplex network (Parallelise version)
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####

## Apply Random Walk with Restart for each node of the network. 
## (Parallelise version)

###### Version using the package doParallel

message("Computing RWR for every network node...")

cl <- parallel::makeForkCluster(mc <- getOption("cl.cores", opt$cores))
doParallel::registerDoParallel(cl)

Results <- foreach(i = 1:length(Allnodes), .packages=c("Matrix") ) %dopar% {
  Random.Walk.Restart.Multiplex.default(AdjMatrixNorm,MultiplexObject,
  Allnodes[i],r=opt$restart,DispResults = "Alphabetic", MeanType="Sum")
}

on.exit(stopCluster(cl))


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### Generation of the output Similarity Matrix
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####

## Generation of the RWR similarity matrix
message("Building Similarity Matrix...")
RWRM_similarity <- matrix(data=0,nrow = numberNodes,ncol = numberNodes)

for (j in seq(numberNodes)){
  RWRM_similarity[,j] <- Results[[j]]$RWRM_Results$Score
}


rownames(RWRM_similarity) <- Allnodes
colnames(RWRM_similarity) <- Allnodes

## Here comes the trick. If we had isolated nodes, which have been removed
## from the analyses because the RWR cannot deal with them, we add them into
## the sim matrix with 1 on the node, 0 elsewhere in the matrix. This only 
## applies to Monoplex networks where the particle can get trapped. 



if (Number_Layers==1){
  numberIso <- length(IsolatedVertex)
  if (numberIso > 0){
    print("Including isolated nodes in the Similarity Matrix...")
    TotalSizeMatrix <-  numberNodes + numberIso
    Index_IsoNodes <- IsolatedVertex
  
    NewMatrix <- matrix(data = 0, nrow =TotalSizeMatrix ,ncol = TotalSizeMatrix)
    rownames(NewMatrix) <- c(rownames(RWRM_similarity),Index_IsoNodes)
    colnames(NewMatrix) <- c(colnames(RWRM_similarity),Index_IsoNodes)
  
    NewMatrix[1:numberNodes,1:numberNodes] <- RWRM_similarity
    NewMatrix[(numberNodes+1):TotalSizeMatrix,(numberNodes+1):TotalSizeMatrix] <- diag(1,numberIso,numberIso)
  
    RWRM_similarity <- NewMatrix
  }
  ## Check how do we want to order
  # NewMatrix <- NewMatrix[order(rownames(NewMatrix)), order(colnames(NewMatrix))] 
  
  # NewMatrix <- NewMatrix[order(as.integer(rownames(NewMatrix))), order(as.integer(colnames(NewMatrix)))] 
}

## check ! Be carefull with this order
if (is.numeric(rownames(RWRM_similarity))){
  RWRM_similarity <- RWRM_similarity[order(as.numeric(rownames(RWRM_similarity))), 
                                     order(as.numeric(colnames(RWRM_similarity)))]
} else {
  RWRM_similarity <- RWRM_similarity[order(rownames(RWRM_similarity)), 
                                     order(colnames(RWRM_similarity))]
}

# as.matrix(RWRM_similarity)
# colSums(RWRM_similarity)
Similarity_outputfile <- paste0(opt$out,".rds")
message(paste0("Saving file ", Similarity_outputfile))
saveRDS(RWRM_similarity,file = Similarity_outputfile)





