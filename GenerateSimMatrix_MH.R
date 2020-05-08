#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### GenerateSimMatrix_MH.R: Script that generates the Similarity matrix (output) 
#### of a given multiplex-heterogeneous network (input) based on the scores of 
#### the RWR (see paper: https://www.ncbi.nlm.nih.gov/pubmed/30020411)
#### been performed on every single node of the network. 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####

#### Usage: 

# Usage: Rscript GenerateSimMatrix_MH.R [options]

#   Options:
#       -n CHARACTER, --network1=CHARACTER
#          Path to the first multiplex network to be used as Input. 
#          It should be a space separated four column file containing 
#          the fields: edge_type, source, target and weight.
#            edge_type source target weight
#             r1     	  n1    n2    1
#             r2        n2    n3    1

#       -m CHARACTER, --network2=CHARACTER
#          Path to the second multiplex network to be used as Input. 
#          It should be a space separated four column file containing 
#          the fields: edge_type, source, target and weight.
#            edge_type source target weight
#             r1     	  n1    n2    1
#             r2        n2    n3    1

#       -b CHARACTER, --bipartite=CHARACTER
#          Path to the bipartite network to be used as Input. 
#          It should be a space separated four column file containing 
#          the fields: edge_type, source, target and weight. Source Nodes
#          should be the ones from the first multiplex and target nodes
#          from the second.
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

setwd(getwd())

## We load the R file containing the associated RWR functions.
source("Functions_RWRMH.R")

## Installation and load of the required R Packages
packages <- c("igraph", "mclust","Matrix","kernlab", "R.matlab","bc3net",
              "optparse","parallel","tidyverse")
ipak(packages)

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### Reading and checking the input arguments
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
## Reading input arguments

#!/usr/bin/env Rscript
print("Reading arguments...")
option_list = list(
  make_option(c("-n", "--network1"), type="character", default=NULL, 
              help="Path to the first multiplex network to be used as Input. 
              It should be a space separated four column file containing 
              the fields: edge_type, source, target and weight.", 
              metavar="character"),
  make_option(c("-m", "--network2"), type="character", default=NULL, 
              help="Path to the second multiplex network to be used as Input. 
              It should be a space separated four column file containing 
              the fields: edge_type, source, target and weight.", 
              metavar="character"),
  make_option(c("-b", "--bipartite"), type="character", default=NULL, 
              help="Path to the bipartite network to be used as Input. 
              It should be a space separated four column file containing 
              the fields: edge_type, source, target and weight. Source Nodes
              should be the ones from the first multiplex and target nodes
              from the second.", 
              metavar="character"),
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

# opt$network1 <- "Networks/m1_toy.txt"
# opt$network2 <- "Networks/m2_toyMulti.txt"
# opt$bipartite <-"Networks/bipartite_toy.txt" 

multiplex1 <- checkNetworks(opt$network1)
multiplex2 <- checkNetworks(opt$network2)
bipartite <- checkNetworks(opt$bipartite)

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

LayersMultiplex1 <- MultiplexToList(multiplex1)
LayersMultiplex2 <- MultiplexToList(multiplex2)

## We create the multiplex objects and multiplex heterogeneous.
MultiplexObject1 <- create.multiplex(LayersMultiplex1)
MultiplexObject2 <- create.multiplex(LayersMultiplex2)
Allnodes1 <- MultiplexObject1$Pool_of_Nodes
Allnodes2 <- MultiplexObject2$Pool_of_Nodes

multiHetObject <- 
      create.multiplexHet(MultiplexObject1,MultiplexObject2,bipartite)

## We have now to compute the transition Matrix to be able to apply RWR

MultiHetTranMatrix <- compute.transition.matrix(multiHetObject)

## Is there any totally isolated node? Convert NAs to 0s
MultiHetTranMatrix[which(is.na(MultiHetTranMatrix))] <- 0

Allnodes <- c(multiHetObject$Multiplex1$Pool_of_Nodes, 
              multiHetObject$Multiplex2$Pool_of_Nodes)
numberNodes <- length(Allnodes)

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### Apply RWR on every node of the multiplex heterogeneous network 
#### (Parallelise version)
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####

## Apply Random Walk with Restart for each node of the network. 
## (Parallelise version)

print("Computing RWR for every network node...")
start_time <- Sys.time()

cl <- makeCluster(mc <- getOption("cl.cores", opt$cores))

Allfuncs <- c("isMultiplexHet", "get.seed.scoresMultiplex", "geometric.mean", 
              "regular.mean", "Random.Walk.Restart.MultiplexHet", "sumValues",
              "get.seed.scores.multHet")

clusterExport(cl=cl, varlist=c("Allnodes1","Allnodes2", "MultiHetTranMatrix",
                               "opt","multiHetObject",Allfuncs))
                               
clusterEvalQ(cl = cl, c("isMultiplex", "get.seed.scores.multHet", 
  "geometric.mean", "regular.mean", "Random.Walk.Restart.Multiplex.default", 
  "Random.Walk.Restart.Multiplex", "sumValues", library("Matrix")))

Results_Multiplex1 <- parLapply(cl, Allnodes1, function (x) 
  Random.Walk.Restart.MultiplexHet(MultiHetTranMatrix,multiHetObject,x,
    Multiplex2_Seeds=c(), r=opt$restart,eta=0.5, DispResults = "Alphabetic",
    MeanType="Sum"))

Results_Multiplex2 <- parLapply(cl, Allnodes2, function (x) 
  Random.Walk.Restart.MultiplexHet(MultiHetTranMatrix,multiHetObject, 
    Multiplex1_Seeds=c(), x, r=opt$restart,eta=0.5, DispResults = "Alphabetic",
    MeanType="Sum"))

stopCluster(cl)
end_time <- Sys.time()

diff <- end_time - start_time
print(paste0("Running Time RWR for every node: ", diff))

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### Generation of the output Similarity Matrix
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####

## Generation of the RWR similarity matrix
print("Building Similarity Matrix...")

RWRMH_similarity1 <- 
  matrix(data=0,nrow = numberNodes, 
         ncol = multiHetObject$Multiplex1$Number_of_Nodes_Multiplex)
RWRMH_similarity2 <- 
  matrix(data=0,nrow = numberNodes,
         ncol = multiHetObject$Multiplex2$Number_of_Nodes_Multiplex)

for (j in seq(multiHetObject$Multiplex1$Number_of_Nodes_Multiplex)){
  RWRMH_similarity1[,j] <- Results_Multiplex1[[j]]$RWRMH_Results$Score 
}

for (j in seq(multiHetObject$Multiplex2$Number_of_Nodes_Multiplex)){
  RWRMH_similarity2[,j] <- Results_Multiplex2[[j]]$RWRMH_Results$Score 
}

RWRMH_similarity <- cbind(RWRMH_similarity1,RWRMH_similarity2)

rownames(RWRMH_similarity) <- Allnodes
colnames(RWRMH_similarity) <- Allnodes


if (is.numeric(rownames(RWRMH_similarity))){
  RWRMH_similarity <- RWRMH_similarity[order(as.numeric(rownames(RWRMH_similarity))), 
                                     order(as.numeric(colnames(RWRMH_similarity)))]
} else {
  RWRMH_similarity <- RWRMH_similarity[order(rownames(RWRMH_similarity)), 
                                     order(colnames(RWRMH_similarity))]
}

### Trick to fix isolated nodes. We check if the particle got trapped in any
### node (The sum over that column will be equal to the restart probability 
### instead of being one)

# We find the column of the isolated node amd we it to 1:
idx <- which(colSums(RWRMH_similarity)== opt$restart)
RWRMH_similarity[idx,idx] <- 1

# as.matrix(RWRMH_similarity)
# colSums(RWRMH_similarity)

Similarity_outputfile <- paste0(opt$out,".rds")
print(paste0("Saving file ", Similarity_outputfile))
saveRDS(RWRMH_similarity,file = Similarity_outputfile)



