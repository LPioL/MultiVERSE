#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### FUNCTIONS to perform RWR-MH 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####

# ipak function: install and load multiple R packages.
# check to see if packages are installed. Install them if they are not, then load 
# them into the R session.

ipak <- function(pkg){
  new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
  if (length(new.pkg)) 
    install.packages(new.pkg, dependencies = TRUE)
  sapply(pkg, require, character.only = TRUE, quietly = TRUE)
}


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####


## 2.- Functions of the RandomWalkMH (Modified to admit weigths and arithmetic 
## mean computation.)

## 2.1.- Create a multiplex object.

isMultiplex <- function (x)
{
  is(x,"Multiplex")
}

isMultiplexHet <- function (x)
{
  is(x,"MultiplexHet")
}

## Add missing nodes in some of the layers.
add.missing.nodes <- function (Layers,Nr_Layers,NodeNames) {
  
  add_vertices(Layers,
               length(NodeNames[which(!NodeNames %in% V(Layers)$name)]),
               name=NodeNames[which(!NodeNames %in%  V(Layers)$name)])
}

## Create a multiplex object
create.multiplex <- function(...){
  UseMethod("create.multiplex")
}


create.multiplex.default <- function(LayersList,...)
{
  
  Number_of_Layers <- length(LayersList)
  SeqLayers <- seq(Number_of_Layers)
  
  if (!all(sapply(SeqLayers, function(x) is.igraph(LayersList[[x]])))){
    stop("Not igraph objects")
  }

  ## We get a pool of nodes (Nodes in any of the layers.)
  Pool_of_Nodes <- 
    sort(unique(unlist(lapply(SeqLayers, function(x) V(LayersList[[x]])$name))))
  
  if (is.numeric(Pool_of_Nodes)){
    Pool_of_Nodes <- sort(as.numeric(Pool_of_Nodes))  
  } else {
    Pool_of_Nodes <- sort(Pool_of_Nodes)
  }
  
  Number_of_Nodes <- length(Pool_of_Nodes)
  
  Layer_List <-
    lapply(LayersList, add.missing.nodes,Number_of_Layers,Pool_of_Nodes)
  

  MultiplexObject <- c(Layer_List,list(Pool_of_Nodes=Pool_of_Nodes,
                                       Number_of_Nodes_Multiplex=Number_of_Nodes, 
                                       Number_of_Layers=Number_of_Layers))
  
  class(MultiplexObject) <- "Multiplex"
  
  return(MultiplexObject)
}

print.Multiplex <- function(x,...)
{
  cat("Number of Layers:\n")
  print(x$Number_of_Layers)
  cat("\nNumber of Nodes:\n")
  print(x$Number_of_Nodes)
  for (i in seq_len(x$Number_of_Layers)){
    cat("\n")
    print(x[[i]])
  }
}

## 2.2.- Adjacency Matrices and Normalization.
compute.adjacency.matrix <- function(x,delta = 0.5)
{
  if (!isMultiplex(x) & !isMultiplexHet(x)) {
    stop("Not a Multiplex or Multiplex Heterogeneous object")
  }
  if (delta > 1 || delta < 0) {
    stop("Delta should be between 0 and 1")
  }
  
  N <- x$Number_of_Nodes_Multiplex
  L <- x$Number_of_Layers
  
  ## We impose delta=0 in the monoplex case.
  if (L==1){
    delta = 0
  }
  
  Layers_Names <- names(x)[seq(L)]
  
  
  ## IDEM_MATRIX.
  Idem_Matrix <- Matrix::Diagonal(N, x = 1)
  
  counter <- 0 
  Layers_List <- lapply(x[Layers_Names],function(x){
    
    counter <<- counter + 1;    
    if (is_weighted(x)) {
      Adjacency_Layer <-  as_adjacency_matrix(x,sparse = TRUE,
                                              attr = "weight")
    } else {
      Adjacency_Layer <-  as_adjacency_matrix(x,sparse = TRUE)
    }
    
    if (is.numeric(rownames(Adjacency_Layer))){
      Adjacency_Layer <- Adjacency_Layer[order(as.numeric(rownames(Adjacency_Layer))),
              order(as.numeric(colnames(Adjacency_Layer)))]
    } else {
      Adjacency_Layer <- Adjacency_Layer[order(rownames(Adjacency_Layer)),
                                         order(colnames(Adjacency_Layer))]
    }
    
    colnames(Adjacency_Layer) <- 
      paste0(colnames(Adjacency_Layer),"_",counter)
    rownames(Adjacency_Layer) <- 
      paste0(rownames(Adjacency_Layer),"_",counter)
    Adjacency_Layer
  })
  
  MyColNames <- unlist(lapply(Layers_List, function (x) unlist(colnames(x))))
  MyRowNames <- unlist(lapply(Layers_List, function (x) unlist(rownames(x))))
  names(MyColNames) <- c()
  names(MyRowNames) <- c()
  SupraAdjacencyMatrix <- (1-delta)*(bdiag(unlist(Layers_List)))
  colnames(SupraAdjacencyMatrix) <-MyColNames
  rownames(SupraAdjacencyMatrix) <-MyRowNames
  
  offdiag <- (delta/(L-1))*Idem_Matrix
  
  i <- seq_len(L)
  Position_ini_row <- 1 + (i-1)*N
  Position_end_row <- N + (i-1)*N
  j <- seq_len(L)
  Position_ini_col <- 1 + (j-1)*N
  Position_end_col <- N + (j-1)*N
  
  for (i in seq_len(L)){
    for (j in seq_len(L)){
      if (j != i){
        SupraAdjacencyMatrix[(Position_ini_row[i]:Position_end_row[i]),
                             (Position_ini_col[j]:Position_end_col[j])] <- offdiag
      }    
    }
  }
  
  SupraAdjacencyMatrix <- as(SupraAdjacencyMatrix, "dgCMatrix")
  return(SupraAdjacencyMatrix)
}

normalize.multiplex.adjacency <- function(x)
{
  if (!is(x,"dgCMatrix")){
    stop("Not a dgCMatrix object of Matrix package")
  }
  
  Adj_Matrix_Norm <- t(t(x)/(Matrix::colSums(x, na.rm = FALSE, dims = 1,
                                             sparseResult = FALSE)))
  
  return(Adj_Matrix_Norm)
}


## 2.3.- Random Walk with Restart and their functions
get.seed.scoresMultiplex <- function(Seeds,Number_Layers,tau) {
  
  Nr_Seeds <- length(Seeds)
  
  Seeds_Seeds_Scores <- rep(tau/Nr_Seeds,Nr_Seeds)
  Seed_Seeds_Layer_Labeled <- 
    paste0(rep(Seeds,Number_Layers),sep="_",rep(seq(Number_Layers), 
        length.out = Nr_Seeds*Number_Layers,each=Nr_Seeds))
  
  Seeds_Score <- data.frame(Seeds_ID = Seed_Seeds_Layer_Labeled,
                            Score = Seeds_Seeds_Scores, stringsAsFactors = FALSE)
  
  return(Seeds_Score)
}

geometric.mean <- function(Scores, L, N) {
  
  FinalScore <- numeric(length = N)
  
  for (i in seq_len(N)){
    FinalScore[i] <- prod(Scores[seq(from = i, to = N*L, by=N)])^(1/L)
  }
  
  return(FinalScore)
}

regular.mean <- function(Scores, L, N) {
  
  FinalScore <- numeric(length = N)
  
  for (i in seq_len(N)){
    FinalScore[i] <- mean(Scores[seq(from = i, to = N*L, by=N)])
  }
  
  return(FinalScore)
}

sumValues <- function(Scores, L, N) {
  
  FinalScore <- numeric(length = N)
  
  for (i in seq_len(N)){
    FinalScore[i] <- sum(Scores[seq(from = i, to = N*L, by=N)])
  }
  
  return(FinalScore)
}

Random.Walk.Restart.Multiplex <- function(...) {
  UseMethod("Random.Walk.Restart.Multiplex")
}

Random.Walk.Restart.Multiplex.default <- 
  function(x, MultiplexObject, Seeds, r=0.7,tau,MeanType="Geometric",
           DispResults="TopScores",...){
    
    ### We control the different values.
    if (!is(x,"dgCMatrix")){
      stop("Not a dgCMatrix object of Matrix package")
    }
    
    if (!isMultiplex(MultiplexObject)) {
      stop("Not a Multiplex object")
    }
    
    L <- MultiplexObject$Number_of_Layers
    N <- MultiplexObject$Number_of_Nodes
    
    Seeds <- as.character(Seeds)
    if (length(Seeds) < 1 | length(Seeds) >= N){
      stop("The length of the vector containing the seed nodes is not 
           correct")
    } else {
      if (!all(Seeds %in% MultiplexObject$Pool_of_Nodes)){
        stop("Some of the seeds are not nodes of the network")
      }
    }
    
    if (r >= 1 || r <= 0) {
      stop("Restart partameter should be between 0 and 1")
    }
    
    if(missing(tau)){
      tau <- rep(1,L)/L
    } else {
      tau <- as.numeric(tau)
      if (sum(tau)/L != 1) {
        stop("The sum of the components of tau divided by the number of 
             layers should be 1")
      }
      }
    
    if(!(MeanType %in% c("Geometric","Arithmetic","Sum"))){
      stop("The type mean should be Geometric, Arithmetic or Sum")
    }
    
    if(!(DispResults %in% c("TopScores","Alphabetic"))){
      stop("The way to display RWRM results should be TopScores or
           Alphabetic")
    }
    
    ## We define the threshold and the number maximum of iterations for
    ## the random walker.
    Threeshold <- 1e-10
    NetworkSize <- ncol(x)
    
    ## We initialize the variables to control the flux in the RW algo.
    residue <- 1
    iter <- 1
    
    ## We compute the scores for the different seeds.
    Seeds_Score <- get.seed.scoresMultiplex(Seeds,L,tau)
    
    ## We define the prox_vector(The vector we will move after the first RWR
    ## iteration. We start from The seed. We have to take in account
    ## that the walker with restart in some of the Seed nodes, depending on
    ## the score we gave in that file).
    prox_vector <- matrix(0,nrow = NetworkSize,ncol=1)
    
    prox_vector[which(colnames(x) %in% Seeds_Score[,1])] <- (Seeds_Score[,2])
    
    prox_vector  <- prox_vector/sum(prox_vector)
    restart_vector <-  prox_vector
    
    while(residue >= Threeshold){
      
      old_prox_vector <- prox_vector
      prox_vector <- (1-r)*(x %*% prox_vector) + r*restart_vector
      residue <- sqrt(sum((prox_vector-old_prox_vector)^2))
      iter <- iter + 1;
    }
    
    NodeNames <- character(length = N)
    Score = numeric(length = N)
    
    rank_global <- data.frame(NodeNames = NodeNames, Score = Score)
    rank_global$NodeNames <- gsub("_1", "", row.names(prox_vector)[seq_len(N)])
    
    if (MeanType=="Geometric"){
      rank_global$Score <- geometric.mean(as.vector(prox_vector[,1]),L,N)    
    } else {
      if (MeanType=="Arithmetic") {
        rank_global$Score <- regular.mean(as.vector(prox_vector[,1]),L,N)    
      } else {
        rank_global$Score <- sumValues(as.vector(prox_vector[,1]),L,N)    
      }
    }
    
    if (DispResults=="TopScores"){
      ## We sort the nodes according to their score.
      Global_results <- 
        rank_global[with(rank_global, order(-Score, NodeNames)), ]
      
      ### We remove the seed nodes from the Ranking and we write the results.
      Global_results <- 
        Global_results[which(!Global_results$NodeNames %in% Seeds),]
    } else {
      Global_results <- rank_global    
    }
    
    rownames(Global_results) <- c()
    
    RWRM_ranking <- list(RWRM_Results = Global_results,Seed_Nodes = Seeds)
    
    class(RWRM_ranking) <- "RWRM_Results"
    return(RWRM_ranking)
    }


print.RWRM_Results <- function(x,...)
{
  cat("Top 10 ranked Nodes:\n")
  print(head(x$RWRM_Results,10))
  cat("\nSeed Nodes used:\n")
  print(x$Seed_Nodes)
}


## 2.4.- Cluster Generation from RWR-M results

ClusterRWSimulation <- function(C,IndividualRWR,NrNodes,AllNodes){
  sum <- numeric(length = NrNodes)
  size <- length(C)
  allIndex <- numeric()
  for (j in seq(size)){
    idx <- which(C[j] == AllNodes)
    sum <- sum + IndividualRWR[[idx]]$RWRM_Results$Score
    allIndex <- c(allIndex,idx)
  }
  
  x <- sum/size
  ClustResults <- data.frame(Nodes = AllNodes, Scores=x, stringsAsFactors = FALSE)
  # ClustResults <- list(Scores = x, Index = allIndex)
  return(ClustResults)
}


###### Specific functions for RWR-MH

## Sanity checks for the input networks
checkNetworks <- function(network){
  if (is.null(network)){
    print_help(opt_parser)
    stop("You need to specify the input network to be used.", call.=FALSE)
  }  
  
  InputNetwork <- read.csv(network, header = FALSE, sep=" ")
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
    print(paste0("Input network with ", Number_Layers, " Layers"))
    ## We check that the weights are numeric. 
    if (!is.numeric(InputNetwork$weight)){
      print_help(opt_parser)
      stop("The weights in the input network should be numeric", 
           call.=FALSE)
    }
  }
  return(InputNetwork)
}

## We transform the input multiplex format to a L-length list of igraphs objects.
## We also scale the weigths of every layer between 1 and the minimun 
## divided by the maximun weight. 
MultiplexToList <- function(network){
  LayersNames <- unique(network$EdgeType)
  Layers <- lapply(LayersNames, function(x) {
    Original_weight <- network$weight[network$EdgeType ==x]
    if (!all(Original_weight)){
      b <- 1
      a <- min(Original_weight)/max(Original_weight)
      range01 <- (b-a)*(Original_weight-min(Original_weight))/
        (max(Original_weight)-min(Original_weight)) + a
      network$weight[network$EdgeType ==x] <- range01
    }
    
    igraph::simplify(graph_from_data_frame(network[network$EdgeType == x,2:4],
                                           directed = FALSE),edge.attr.comb=mean)
  })
  names(Layers) <-LayersNames
  return(Layers)
}

create.multiplexHet <- function(...) {
  UseMethod("create.multiplexHet")
}

create.multiplexHet.default  <- function(MultiObject1, MultiObject2, 
                                         BipartiteNetwork,...)
{
  
  Allnodes1 <- MultiObject1$Pool_of_Nodes
  Allnodes2 <- MultiObject2$Pool_of_Nodes
  
  bipartiteNodesNetwork1 <- unique(c(as.character(BipartiteNetwork$source)))
  bipartiteNodesNetwork2 <- unique(c(as.character(BipartiteNetwork$target)))
  
  if (!all(bipartiteNodesNetwork1 %in% Allnodes1)){
    stop("Some of the source nodes of the bipartite are not present in
         the first network")
  } 
  
  if (!all(bipartiteNodesNetwork2 %in% Allnodes2)){
    stop("Some of the target nodes of the bipartite are not present in
         the secondnetwork")
  }
  
  ## Multiplex graph features
  NumberNodes1 <- MultiObject1$Number_of_Nodes
  NumberLayer1 <- MultiObject1$Number_of_Layers
  
  NumberNodes2 <- MultiObject2$Number_of_Nodes
  NumberLayer2 <- MultiObject2$Number_of_Layers
  
  message("Generating bipartite matrix...")
  Bipartite_Matrix <- 
    get.bipartite.graph(Allnodes1,Allnodes2,BipartiteNetwork,NumberNodes1, 
                        NumberNodes2)
  
  message("Expanding bipartite matrix to fit the multiplex network...")
  Supra_Bipartite_Matrix <- expand.bipartite.graph(NumberNodes1,NumberLayer1,
                                                   NumberNodes2,NumberLayer2,Bipartite_Matrix)
  
  Multiplex_HetObject <- list(Multiplex1 = MultiObject1,
                              Multiplex2 = MultiObject2,
                              BipartiteNetwork = Supra_Bipartite_Matrix)
  
  class(Multiplex_HetObject) <- "MultiplexHet"
  return(Multiplex_HetObject)
  }

print.MultiplexHet <- function(x,...)
{
  cat("Network Object 1:\n")
  print(x$Multiplex1)
  cat("\nNetwork Object 2:\n")
  print(x$Multiplex2)
  cat("\nSupra Bipartite Network:\n")
  print(x$BipartiteNetwork)
}


## Bipartite graph construction.
get.bipartite.graph <- function(Names_Mul1, Names_Mul2, BipartiteNetwork, 
                                Number_Nodes_1,Number_Nodes_2){
  
  Bipartite_matrix <- Matrix(data=0, nrow=Number_Nodes_1, ncol=Number_Nodes_2)
  Names_Mul1_order <- sort(Names_Mul1)
  Names_Mul2_order <- sort(Names_Mul2)
  rownames(Bipartite_matrix) <- Names_Mul1_order
  colnames(Bipartite_matrix) <- Names_Mul2_order
  
  for (i in seq(nrow(BipartiteNetwork))){
    Bipartite_matrix[BipartiteNetwork$source[i],BipartiteNetwork$target[i]] <- 
      BipartiteNetwork$weight[i]
  }
  
  return(Bipartite_matrix)
}


## Fitting the bipartite graph to the multiplex networks.
expand.bipartite.graph <- 
  function(Number_Nodes_1,Number_Layers_1,Number_Nodes_2,
           Number_Layers_2,Bipartite_matrix){
    
    Supra_Bipartite_Matrix <- 
      do.call(rbind, replicate(Number_Layers_1,Bipartite_matrix,simplify=FALSE))
    
    rownames(Supra_Bipartite_Matrix) <- 
      paste0(rownames(Bipartite_matrix), sep="_",rep(seq(Number_Layers_1),
                                                     each=Number_Nodes_1))
    
    
    Supra_Bipartite_Matrix <- 
      do.call(cbind, replicate(Number_Layers_2,Supra_Bipartite_Matrix,
                               simplify=FALSE))
    
    colnames(Supra_Bipartite_Matrix) <- 
      paste0(colnames(Bipartite_matrix), sep="_",rep(seq(Number_Layers_2),
                                                     each=Number_Nodes_2))
    
    return(Supra_Bipartite_Matrix)
  }


### Transition functions
get.transition.multiplex1.multiplex2 <- 
  function(Number_Nodes_Multiplex1, Number_Layers1,Number_Nodes_Multiplex2,
           Number_Layers2, SupraBipartiteMatrix,lambda){
    
    TransitionMat_Multiplex1_Multiplex2 <- 
      Matrix(0, nrow=Number_Nodes_Multiplex1*Number_Layers1,
             ncol=Number_Nodes_Multiplex2*Number_Layers2,sparse = TRUE)
    
    colnames(TransitionMat_Multiplex1_Multiplex2) <- 
      colnames(SupraBipartiteMatrix)
    rownames(TransitionMat_Multiplex1_Multiplex2) <- 
      rownames(SupraBipartiteMatrix)
    
    Col_Sum_Bipartite <- 
      Matrix::colSums (SupraBipartiteMatrix, na.rm = FALSE, dims = 1, 
                       sparseResult = FALSE)
    
    m <- lambda * t(t(SupraBipartiteMatrix) / Col_Sum_Bipartite)
    idx <- Col_Sum_Bipartite != 0
    TransitionMat_Multiplex1_Multiplex2[,idx] = m[,idx]
    
    return(TransitionMat_Multiplex1_Multiplex2)
  }

get.transition.multiplex2.multiplex1 <- 
  function(Number_Nodes_Multiplex1, Number_Layers1,Number_Nodes_Multiplex2,
           Number_Layers2,SupraBipartiteMatrix,lambda){
    
    TransitionMat_Multiplex2_Multiplex1 <- 
      Matrix(0,nrow=Number_Nodes_Multiplex2*Number_Layers2, 
             ncol=Number_Nodes_Multiplex1*Number_Layers1,sparse = TRUE)
    
    colnames(TransitionMat_Multiplex2_Multiplex1) <- 
      rownames(SupraBipartiteMatrix)
    rownames(TransitionMat_Multiplex2_Multiplex1) <- 
      colnames(SupraBipartiteMatrix)
    
    Row_Sum_Bipartite <- 
      Matrix::rowSums (SupraBipartiteMatrix, na.rm = FALSE, dims = 1,
                       sparseResult = FALSE)
    
    m <- lambda * t((SupraBipartiteMatrix) / Row_Sum_Bipartite)
    idx <- Row_Sum_Bipartite != 0
    TransitionMat_Multiplex2_Multiplex1[,idx] = m[,idx]
    
    return(TransitionMat_Multiplex2_Multiplex1)
  }


get.transition.multiplex <- 
  function(Number_Nodes,Number_Layers, lambda,SupraAdjacencyMatrix,
           SupraBipartiteMatrix) {
    
    Transition_Multiplex_Network <- 
      Matrix(0, nrow=Number_Nodes*Number_Layers,
             ncol=Number_Nodes*Number_Layers,sparse = TRUE)
    
    rownames(Transition_Multiplex_Network) <- rownames(SupraAdjacencyMatrix)
    colnames(Transition_Multiplex_Network) <- colnames(SupraAdjacencyMatrix)
    
    Col_Sum_Multiplex <- 
      Matrix::colSums(SupraAdjacencyMatrix,na.rm=FALSE, dims=1, 
                      sparseResult=FALSE)
    Row_Sum_Bipartite <- 
      Matrix::rowSums (SupraBipartiteMatrix, na.rm = FALSE, dims = 1,
                       sparseResult = FALSE)
    
    idx <- Row_Sum_Bipartite != 0
    Transition_Multiplex_Network[,idx] <- 
      ((1-lambda)*t(t(SupraAdjacencyMatrix[,idx])/Col_Sum_Multiplex[idx]))
    
    Transition_Multiplex_Network[,!idx] <-
      t(t(SupraAdjacencyMatrix[,!idx]) / Col_Sum_Multiplex[!idx])
    
    return(Transition_Multiplex_Network)
  }

#### Transition Matrices!! 

compute.transition.matrix <- function(x,lambda = 0.5, delta1=0.5,
                                      delta2=0.5)
{
  if (!isMultiplexHet(x)) {
    stop("Not a Multiplex Heterogeneous object")
  }
  
  if (delta1 > 1 || delta1 <= 0) {
    stop("Delta should be between 0 and 1")
  }
  
  if (delta2 > 1 || delta2 <= 0) {
    stop("Delta should be between 0 and 1")
  }
  
  if (lambda > 1 || lambda <= 0) {
    stop("Lambda should be between 0 and 1")
  }
  
  NumberNodes1 <- x$Multiplex1$Number_of_Nodes_Multiplex
  NumberLayers1 <- x$Multiplex1$Number_of_Layers
  
  NumberNodes2 <- x$Multiplex2$Number_of_Nodes_Multiplex
  NumberLayers2 <- x$Multiplex2$Number_of_Layers
  
  SupraBipartiteMatrix <- x$BipartiteNetwork
  
  message("Computing adjacency matrix of the first input network...")
  AdjMatrix_Multiplex1 <- compute.adjacency.matrix(x$Multiplex1,delta1)
  
  message("Computing adjacency matrix of the second input network...")
  AdjMatrix_Multiplex2 <- compute.adjacency.matrix(x$Multiplex2,delta2)
  
  ## Transition Matrix for the inter-subnetworks links
  message("Computing inter-subnetworks transitions...")
  Transition_Multiplex1_Multiplex2 <- 
    get.transition.multiplex1.multiplex2(NumberNodes1,NumberLayers1,
                                         NumberNodes2,NumberLayers2,SupraBipartiteMatrix,lambda)
  
  Transition_Multiplex2_Multiplex1 <- 
    get.transition.multiplex2.multiplex1(NumberNodes1,NumberLayers1,
                                         NumberNodes2, NumberLayers2, SupraBipartiteMatrix,lambda)
  
  ## Transition Matrix for the intra-subnetworks links
  message("Computing intra-subnetworks transitions...")
  Transition_Multiplex_Network1 <- 
    get.transition.multiplex(NumberNodes1, NumberLayers1, lambda, 
                             AdjMatrix_Multiplex1,SupraBipartiteMatrix)
  Transition_Multiplex_Network2 <- 
    get.transition.multiplex(NumberNodes2,NumberLayers2,lambda,
                             t(AdjMatrix_Multiplex2),t(SupraBipartiteMatrix))
  
  ## We generate the global transiction matrix and we return it.
  message("Combining inter e intra layer probabilities into the global 
          Transition Matix")
  Transition_Multiplex_Heterogeneous_Matrix_1 <-
    cbind(Transition_Multiplex_Network1, Transition_Multiplex1_Multiplex2)
  Transition_Multiplex_Heterogeneous_Matrix_2 <-
    cbind(Transition_Multiplex2_Multiplex1, Transition_Multiplex_Network2)
  Transition_Multiplex_Heterogeneous_Matrix <-
    rbind(Transition_Multiplex_Heterogeneous_Matrix_1,
          Transition_Multiplex_Heterogeneous_Matrix_2)
  
  return(Transition_Multiplex_Heterogeneous_Matrix)
}


#### Computing the scores for the different seed nodes

get.seed.scores.multHet <- 
  function(Multiplex1_Seed_Nodes,Multiplex2_Seed_Nodes,eta,L1,L2,tau1,tau2) {
    
    n <- length(Multiplex1_Seed_Nodes)
    m <- length(Multiplex2_Seed_Nodes)
    
    if ((n != 0 && m!= 0)){
      
      Seed_Multiplex1_Layer_Labeled <- paste0(rep(Multiplex1_Seed_Nodes,L1),
                                              sep="_",rep(seq(L1), length.out = n*L1,each=n))
      
      Seed_Multiplex2_Layer_Labeled <- paste0(rep(Multiplex2_Seed_Nodes,L2),
                                              sep="_",rep(seq(L2), length.out = m*L2,each=m))
      
      Seeds_Multiplex1_Scores <- rep(((1-eta) * tau1)/n,n)
      Seeds_Multiplex2_Scores <- rep((eta*tau2)/m,m)
      
    } else {
      eta <- 1
      if (n == 0){
        Seed_Multiplex1_Layer_Labeled <- character()
        Seeds_Multiplex1_Scores <- numeric()
        Seed_Multiplex2_Layer_Labeled <- 
          paste0(rep(Multiplex2_Seed_Nodes,L2), sep="_",rep(seq(L2),
                                                            length.out = m*L2,each=m))
        Seeds_Multiplex2_Scores <- rep(tau2/m,m)
      } else {
        Seed_Multiplex1_Layer_Labeled <- 
          paste0(rep(Multiplex1_Seed_Nodes,L1), sep="_",rep(seq(L1),
                                                            length.out = n*L1,each=n))
        Seeds_Multiplex1_Scores <- rep(tau1/n,n)
        Seed_Multiplex2_Layer_Labeled <- character()
        Seeds_Multiplex2_Scores <- numeric()
      }
    }
    
    ## We prepare a data frame with the seeds.
    Seeds_Score <- data.frame(Seeds_ID = 
                                c(Seed_Multiplex1_Layer_Labeled,Seed_Multiplex2_Layer_Labeled),
                              Score = c(Seeds_Multiplex1_Scores, Seeds_Multiplex2_Scores),
                              stringsAsFactors = FALSE)
    
    return(Seeds_Score)
  }


Random.Walk.Restart.MultiplexHet <- 
  function(x, MultiplexHet_Object, Multiplex1_Seeds,Multiplex2_Seeds,
           r=0.7,tau1,tau2,eta=0.5,MeanType="Geometric",
           DispResults="TopScores",...){
    
    ## We control the different values.
    if (!"dgCMatrix" %in% class(x)){
      stop("Not a dgCMatrix object of Matrix package")
    }
    
    if (!isMultiplexHet(MultiplexHet_Object)) {
      stop("Not a Multiplex Heterogeneous object")
    }
    
    NumberLayers1 <- MultiplexHet_Object$Multiplex1$Number_of_Layers
    NumberNodes1 <- MultiplexHet_Object$Multiplex1$Number_of_Nodes_Multiplex
    NumberLayers2 <- MultiplexHet_Object$Multiplex2$Number_of_Layers
    NumberNodes2 <- MultiplexHet_Object$Multiplex2$Number_of_Nodes_Multiplex
    
    All_nodes_Multiplex1 <- MultiplexHet_Object$Multiplex1$Pool_of_Nodes
    All_nodes_Multiplex2 <- MultiplexHet_Object$Multiplex2$Pool_of_Nodes
    
    MultiplexSeeds1 <- as.character(Multiplex1_Seeds)
    MultiplexSeeds2 <- as.character(Multiplex2_Seeds)
    
    if (length(MultiplexSeeds1) < 1 & length(MultiplexSeeds2) < 1){
      stop("You did not provided any seeds")
    } else {
      if (length(MultiplexSeeds1) >= NumberNodes1 | length(MultiplexSeeds2) >= NumberNodes2){
        stop("The length of some of the vectors containing the seed nodes 
             is not correct")
      }  else {
        if (!all(MultiplexSeeds1 %in% All_nodes_Multiplex1)){
          stop("Some of the  input seeds are not nodes of the first 
               input network")
        } else {
          if (!all(All_nodes_Multiplex2 %in% All_nodes_Multiplex2)){
            stop("Some of the inputs seeds are not nodes of the second
                 input network")
          }
          }
        }
    }
    
    if (r >= 1 || r <= 0) {
      stop("Restart partameter should be between 0 and 1")
    }
    
    if (eta >= 1 || eta <= 0) {
      stop("Eta partameter should be between 0 and 1")
    }
    
    if(missing(tau1)){
      tau1 <- rep(1,NumberLayers1)/NumberLayers1
    } else {
      tau1 <- as.numeric(tau1)
      if (sum(tau1)/NumberLayers1 != 1) {
        stop("The sum of the components of tau divided by the number of 
             layers should be 1")}
      }
    
    if(missing(tau2)){
      tau2 <- rep(1,NumberLayers2)/NumberLayers2
    } else {
      tau1 <- as.numeric(tau2)
      if (sum(tau2)/NumberLayers2 != 1) {
        stop("The sum of the components of tau divided by the number of 
             layers should be 1")}
      }
    
    if(!(MeanType %in% c("Geometric","Arithmetic","Sum"))){
      stop("The type mean should be Geometric, Arithmetic or Sum")
    }
    
    if(!(DispResults %in% c("TopScores","Alphabetic"))){
      stop("The way to display RWRM results should be TopScores or
           Alphabetic")
    }
    
    ## We define the threshold and the number maximum of iterations
    ## for the random walker.
    Threeshold <- 1e-10
    NetworkSize <- ncol(x)
    
    ## We initialize the variables to control the flux in the RW algo.
    residue <- 1
    iter <- 1
    
    ## We compute the scores for the different seeds.
    Seeds_Score <- 
      get.seed.scores.multHet(Multiplex1_Seeds, Multiplex2_Seeds,eta,
                              NumberLayers1,NumberLayers2,tau1,tau2)
    
    ## We define the prox_vector(The vector we will move after the first
    ## RWR iteration. We start from The seed. We have to take in account
    ## that the walker with restart in some of the Seed genes,
    ## depending on the score we gave in that file).
    prox_vector <- matrix(0,nrow = NetworkSize,ncol=1)
    
    prox_vector[which(colnames(x) %in% Seeds_Score[,1])] <- (Seeds_Score[,2])
    
    prox_vector  <- prox_vector/sum(prox_vector)
    restart_vector <-  prox_vector
    
    while(residue >= Threeshold){
      
      old_prox_vector <- prox_vector
      prox_vector <- (1-r)*(x %*% prox_vector) + r*restart_vector
      residue <- sqrt(sum((prox_vector-old_prox_vector)^2))
      iter <- iter + 1;
    }
    
    IndexSep <- NumberNodes1*NumberLayers1
    prox_vector_1 <- prox_vector[1:IndexSep,]
    prox_vector_2 <- prox_vector[(IndexSep+1):nrow(prox_vector),]
    
    NodeNames1 <- 
      gsub("_1", "",names(prox_vector_1)[seq_len(NumberNodes1)])
    NodeNames2 <- 
      gsub("_1", "",names(prox_vector_2)[seq_len(NumberNodes2)])
    
    
    if (MeanType=="Geometric"){
      rank_global1 <- geometric.mean(prox_vector_1,NumberLayers1,NumberNodes1)    
      rank_global2 <- geometric.mean(prox_vector_2,NumberLayers2,NumberNodes2)   
    } else {
      if (MeanType=="Arithmetic") {
        rank_global1 <- regular.mean(prox_vector_1,NumberLayers1,NumberNodes1)    
        rank_global2 <- regular.mean(prox_vector_2,NumberLayers2,NumberNodes2)     
      } else {
        rank_global1 <- sumValues(prox_vector_1,NumberLayers1,NumberNodes1)    
        rank_global2 <- sumValues(prox_vector_2,NumberLayers2,NumberNodes2) 
      }
    }
    
    
    
    Global_results <- data.frame(NodeNames = c(NodeNames1,NodeNames2), 
                                 Score = c(rank_global1,rank_global2))
    
    if (DispResults=="TopScores"){
      ## We sort the nodes according to their score.
      Global_results <- 
        Global_results[with(Global_results, order(-Score, NodeNames)), ]
      
      ### We remove the seed nodes from the Ranking and we write the results.
      Global_results <- 
        Global_results[which(!Global_results$NodeNames %in% Seeds),]
    } else {
      Global_results <- Global_results    
    }
    
    rownames(Global_results) <- c()
    
    RWRMH_ranking <- list(RWRMH_Results = Global_results,
                          Seed_Nodes = c(Multiplex1_Seeds,Multiplex2_Seeds))
    
    return(RWRMH_ranking)
    }
