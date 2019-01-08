"""
Graph Mining - ALTEGRAD - Dec 2018
"""

# Import modules
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

import os
os.chdir("C:/Users/utilisateur/Desktop/LAST_YEAR/AlteGrad/ALTEGRAD/LAB6")


############## Question 1
# Load the graph into an undirected NetworkX graph

##################
# your code here #
G = nx.read_edgelist("./datasets/CA-HepTh.txt", comments='#', delimiter='\t', nodetype=int, create_using=nx.Graph())
##################


############## Question 2
# Network Characteristics
##################
# your code here #
print('the number of nodes is: ',G.number_of_nodes())
print('the number of edges is: ', G.number_of_edges())
print('the number of connectesd components is: ', nx.number_connected_components(G))

GCC = max(nx.connected_component_subgraphs(G), key=len)
perc_nodes = 100 * GCC.number_of_nodes()/G.number_of_nodes()
perc_edges = 100 * GCC.number_of_edges()/G.number_of_edges()

print('the GCC captures %.2f nodes of the initial graph'%perc_nodes)
print('the GCC captures %.2f edges of the initial graph'%perc_edges)
##################



############## Question 3
# Analysis of degree distribution


##################
# your code here #
degree_sequence = [d for n,d in G.degree() ]

print('the minimum degree is :', np.min(degree_sequence))
print('the maximum degree is :', np.max(degree_sequence))
print('the average degree is :', np.mean(degree_sequence))


y = nx.degree_histogram(G)
plt.plot(y, 'b-', marker='o')
plt.ylabel('Frequency')
plt.xlabel('Degree')
plt.show()

plt.loglog(y, 'b-', marker='o')
plt.ylabel('Frequency')
plt.xlabel('Degree')
plt.show()



##################