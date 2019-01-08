"""
Graph Mining - ALTEGRAD - Dec 2018
"""
import os
os.chdir("C:/Users/utilisateur/Desktop/LAST_YEAR/AlteGrad/ALTEGRAD/LAB6")


import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from random import randint
from sklearn.cluster import KMeans
from networkx.algorithms.community import greedy_modularity_communities


# Load the graph into an undirected NetworkX graph
G = nx.read_edgelist("./datasets/CA-HepTh.txt", comments='#', delimiter='\t', nodetype=int, create_using=nx.Graph())
                
                     
# Get giant connected component (GCC)
GCC = max(nx.connected_component_subgraphs(G), key=len)


############## Question 1
# Implement and apply spectral clustering
def spectral_clustering(G, k, seed=1234):
    L = nx.normalized_laplacian_matrix(G).astype(float) # Normalized Laplacian

    # Calculate k smallest in magnitude eigenvalues and corresponding eigenvectors of L
    # hint: use eigs function of scipy

    ##################
    # your code here #
    eigval,eigvec = eigs(L)
    ##################

    eigval = eigval.real # Keep the real part
    eigvec = eigvec.real # Keep the real part
    
    idx = eigval.argsort() # Get indices of sorted eigenvalues
    eigvec = eigvec[:,idx] # Sort eigenvectors according to eigenvalues
    
    # Perform k-means clustering (store in variable "membership" the clusters to which points belong)
    # hint: use KMeans class of scikit-learn
    

    ##################
    # your code here #
    kmeans = KMeans(n_clusters=k, random_state = seed)
    kmeans.fit(eigvec[:,:k])
    ##################

    # Create a dictionary "clustering" where keys are nodes and values the clusters to which the nodes belong
    
    ##################
    # your code here #
    clusters = kmeans.predict(eigvec[:,:k])
    clustering = {n:c for n,c in zip(GCC.nodes(), clusters)}
    ##################

    return clustering
	
# Apply spectral clustering to the CA-HepTh dataset
clustering = spectral_clustering(G=GCC, k=60, seed=123)


# sanity check
GCC.number_of_nodes() == len(clustering)


############## Question 2
# Implement modularity and compute it for two clustering results

# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    n_clusters = len(set(clustering.values()))
    modularity = 0 # Initialize total modularity value
    m = G.number_of_edges()
    #Iterate over all clusters
    for i in range(n_clusters):
        node_list = [n for n,v in clustering.items() if v == i] # Get the nodes that belong to the i-th cluster
        subG = G.subgraph(node_list) # get subgraph that corresponds to current cluster

        # Compute contribution of current cluster to modularity as in equation 1

        ##################
        # your code here #
        lc = subG.number_of_edges()
        degrees = [v for k,v in subG.degree]   
        dc = sum(degrees)
        modularity += lc/m -(dc/(2*m))**2
        ##################		
        
    return modularity
	
print("Modularity Spectral Clustering: ", modularity(GCC, clustering))

# Implement random clustering
k = 60
r_clustering = dict()

for node in GCC.nodes:
    r_clustering[node] = np.random.randint(0, k)

# Partition randomly the nodes in k clusters (store the clustering result in the "r_clustering" dictionary)
# hint: use randint function

##################
# your code here #
##################  


print("Modularity Random Clustering: ", modularity(GCC, r_clustering))


############## Question 3
# Run Clauset-Newman-Moore algorithm and compute modularity

# Partition graph using the Clauset-Newman-Moore greedy algorithm
# hint: use the greedy_modularity_communities function. The function returns sets of nodes, one for each community.
# Create a dictionary "clustering_cnm" keyed by node to the cluster to which the node belongs

##################
# your code here #
partition = greedy_modularity_communities(GCC)
clustering_cnm = dict()
for c in range(len(partition)):
    for node in partition[c]:
        clustering_cnm [node] = c

##################  

print("Modularity Clauset-Newman-Moore algorithm: ", modularity(GCC, clustering_cnm))


