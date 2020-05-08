#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### EdgelistToMultiplex.R: Script that takes as input monoplex networks
#### in edgelist format and converts it to multiplex format (output):
#### 4 columns: edge_type, source, target, weight"
####            edge_type source target weight
####            r1     	  n1    n2    1
####            r2        n2    n3    1
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####

#### Usage: 

# Usage: Rscript EdgelistToMultiplex.R [options]

#   Options:
#       -n CHARACTER, --networks=CHARACTER
#          Path to the monoplex networks to be used as Inputs. They should be in  
#          edgelist format, with two or three column depending if edges are
#          weighted : source node, target node, weight (optional). If more than 
#          one network is provided, they should be separated by commas. 
#          
#         source target weight
#             n1    n2    1
#             n2    n3    1

#       -e CHARACTER, --edgeTypes=CHARACTER
#       Names for the different layers (Usually the type of edge described by
#       the interactions). If more than one network is provided, they should be 
#       separated by commas. 

#       -o CHARACTER, --out=CHARACTER
#           Name for the output file (The resulting multiplex network)

#       -h, --help
#           Show this help message and exit

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### Loading functions and external packages
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####

## We clean the workspace
rm(list=ls());cat('\014');if(length(dev.list()>0)){dev.off()}

setwd(getwd())

# ipak function: install and load multiple R packages.
# check to see if packages are installed. Install them if they are not, then load 
# them into the R session.

ipak <- function(pkg){
  new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
  if (length(new.pkg)) 
    install.packages(new.pkg, dependencies = TRUE)
  sapply(pkg, require, character.only = TRUE, quietly = TRUE)
}

## Installation and load of the required R Packages
packages <- c("optparse","tidyverse")
ipak(packages)


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### Reading and checking the input arguments
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####

## Reading input arguments

#!/usr/bin/env Rscript
print("Reading arguments...")
option_list = list(
  make_option(c("-n", "--networks"), type="character", default=NULL, 
    help="Path to the monoplex networks to be used as Inputs. They should be in  
    edgelist format, with two or three column depending if edges are
    weighted : source node, target node, weight (optional). If more than 
    one network is provided, they should be separated by commas. 
         source target weight
             n1    n2    1
             n2    n3    1", metavar="character"),
  make_option(c("-e", "--edgeTypes"), type="character", default=NULL, 
    help="Names for the different layers (Usually the type of edge described by
    the interactions). If more than one network is provided, they should be 
    separated by commas. ", metavar="character"),
  make_option(c("-o", "--out"), type="character", default=NULL, 
    help="Name and Path for the output file (The resulting multiplex network)",
    metavar="character")
); 

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

## Checking if the arguments are valid. Otherwise, stop the execution and
## return an error

if (is.null(opt$networks)){
  print_help(opt_parser)
  stop("You need to specify input network(s) to be used.", call.=FALSE)
}  

PathToNetworks <- unlist(strsplit(opt$networks,","))
InputNetworks <- lapply(PathToNetworks, function(x){
  read.csv(x,header = FALSE, sep=" ")
})

NumberLayers <- length(InputNetworks)


if (is.null(opt$edgeTypes)){
  edgeTypes <- seq(NumberLayers)  
} else {
  edgeTypes <- unlist(strsplit(opt$edgeTypes,","))
  if (length(edgeTypes) != NumberLayers){
    stop("The number of names for the edges should be the same than
         the number of input networks.", call.=FALSE)  
  }
}

if (is.null(opt$out)){
  print_help(opt_parser)
  stop("You need to specify a name to be used to generate the output files.", 
       call.=FALSE)
}

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### Input file transformation towards the output file
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####

print("Transforming input networks to multiplex output format...")

InputNetworksReady <- lapply(seq(NumberLayers), function(x){
  if (ncol(InputNetworks[[x]])==2){
    colnames(InputNetworks[[x]]) <- c("source","target")
    mutate(InputNetworks[[x]], weights = 1) %>% 
    mutate(edgeType = edgeTypes[x]) %>%
    select(edgeType,source,target,weights)  
  } else {
    if (ncol(InputNetworks[[x]])==3){
      colnames(InputNetworks[[x]]) <- c("source","target","weights")
      mutate(InputNetworks[[x]], edgeType = edgeTypes[x]) %>%
        select(edgeType,source,target,weights)  
    } else {
      stop("The number of names for the edges should be the same than
         the number of input networks.", call.=FALSE)
    }
  }
})

MultiplexNetworkFormat <-do.call("rbind", InputNetworksReady)

print(paste0("Writing output with multiplex network format containing ", 
              NumberLayers," Layer(s)"))

write.table(MultiplexNetworkFormat, file = opt$out,
            row.names = FALSE, col.names = FALSE, quote = FALSE)
