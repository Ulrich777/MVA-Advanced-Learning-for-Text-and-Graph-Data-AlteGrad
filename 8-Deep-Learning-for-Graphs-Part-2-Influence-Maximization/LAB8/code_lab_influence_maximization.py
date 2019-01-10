import os
os.chdir("C:/Users/utilisateur/Desktop/LAST_YEAR/AlteGrad/ALTEGRAD/LAB8/data")
#os.chdir("/Data/Digg")

import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import ndlib.models.epidemics.ThresholdModel as th





"""
Part 1
"""
G = nx.read_weighted_edgelist("digg_friends_subset.edgelist",create_using=nx.DiGraph())

#--------------------------------- 1.1
top = 20
#indegree = dict(G.in_degree())
indegree = sorted(dict(G.in_degree()).items(), key=lambda x:-x[1])
degree = sorted(dict(G.degree()).items(), key=lambda x:-x[1])
top_indegree = [x[0] for i,x in enumerate(indegree) if i<top]
top_degree = [x[0] for i,x in enumerate(degree) if i<top]

#--------------------------------- 1.2
pg = sorted(nx.pagerank(nx.to_undirected(G)).items(), key=lambda x:-x[1])
dipg = sorted(nx.pagerank(G).items(), key=lambda x:-x[1])

top_pg = [x[0] for i,x in enumerate(pg) if i<top]
top_dipg = [x[0] for i,x in enumerate(dipg) if i<top]
#--------------------------------- 1.3
kcores = sorted(nx.core_number(G).items(), key=lambda x:-x[1])
top_kcores = [x[0] for i,x in enumerate(kcores) if i<top]


#--------------------------------- 1.4
def clean_graph(G,threshold=300):
    to_plot = G.copy()
    
    to_remove = []
    for node, degree in G.degree():
        if degree<threshold:
            to_remove.append(node)
    
    to_plot.remove_nodes_from(to_remove)
    
    print ("Removed "+ str(len(to_remove)) + " of "+ str(len (G.nodes())) + " nodes ") 
    
    return to_plot

to_plot = clean_graph(G)
len(to_plot.edges())

#------------ Filter the top 10 to keep nodes inside the plot
plt_kcores = [node for node in top_kcores if node in to_plot.nodes()][:10]
plt_indegree = [node for node in top_indegree if node in to_plot.nodes()][:10]
plt_dipg = [node for node in top_dipg if node in to_plot.nodes()][:10]


#------------ Plot the network
pos = nx.spring_layout(to_plot)
nx.draw(to_plot,pos,nodelist= to_plot.nodes(),alpha=0.7, node_size=2,node_color="black",arrows =False,width=0.001)

#------------ Plot the chosen nodes with different colors
_ = nx.draw_networkx_nodes(to_plot,pos,nodelist=plt_dipg,node_color='b',node_size=20)
_ = nx.draw_networkx_nodes(to_plot,pos,nodelist=plt_indegree,node_color='g',node_size=20)
_ = nx.draw_networkx_nodes(to_plot,pos,nodelist=plt_kcores,node_color='r',node_size=20)



"""
Part 2
"""
#------------------------------- 1
#2 Run simulation to evaluate the chosen seeds
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics.ThresholdModel as th


def simulate_spreading(G,seed_set,num_steps=3,threshold = 0.01):
    """
    Given the graph and the seed set, compute the number of infected nodes after the end of a spreading process
    """
    th_model = th.ThresholdModel(G)
    config = mc.Configuration()
    config.add_model_initial_configuration("Infected", seed_set)
    
    for node in G.nodes():
        config.add_node_configuration("threshold", node, threshold)
        
    th_model.set_initial_status(config)
    iterations = th_model.iteration_bunch(num_steps)
   
    
    return iterations[num_steps-1]["node_count"][1]

    
# Call simulate_spreading for all sets
count_indegree = simulate_spreading(G, top_indegree)
count_degree = simulate_spreading(G, top_degree)
count_pg = simulate_spreading(G, top_pg)
count_dipg = simulate_spreading(G, top_dipg)
count_kcores = simulate_spreading(G, top_kcores)

#------------------------------- 2
# Load diffusion cascades 
f = open("digg_votes_subset.txt","r")
node_cascades = {}
for l in f:
    parts = l.replace("\n","").split(",")
    starter = parts[0]
    if(starter not in node_cascades):
        node_cascades[starter] = []
    node_cascades[starter].append(",".join(parts[1:len(parts)]))
f.close()

    
# Compute DNI
def dni(node_cascades,seed_set):
    """
    Measure the number of distinct nodes in the cascades started by the seed set
    """
    influence_spread = set()  
    
    for node in seed_set:
        try:
            cascade = node_cascades[node]
        except:
            cascade = set()
        influence_spread = influence_spread.union(set(cascade))
  
    
    
    return len(influence_spread)

        
#Call dni for all sets
dni_indegree = dni(node_cascades, top_indegree)
dni_degree = dni(node_cascades, top_degree)
dni_pg = dni(node_cascades, top_pg)
dni_dipg = dni(node_cascades, top_dipg)
dni_kcores = dni(node_cascades, top_kcores)


"""
Part 3
"""
# Implement Greedy
def greedy_algorithm(G, selected_nodes, size):
    """
    Greedy influence maximization algorithm (Kempe et all at 2003)
    """
    s = set()
    not_visited = selected_nodes.copy()
    while(len(s)<size):
        
        maxi = 0.
        for v in  not_visited:
            s_ = s.union(set([v]))
            spread = simulate_spreading(G, s_)
            if spread>maxi:
                maxi = spread
                v_opt = v
        s = s.union(set([v_opt]))
        not_visited.remove(v_opt)
        print(len(s))
    return s

to_plot = clean_graph(G,850)


top=10

greedy_set = greedy_algorithm(G,list(to_plot.nodes()),top)


#------- Seed sets for comparison
top_indegree = top_indegree[:top]
top_degree = top_degree[:top]
top_pg = top_pg[:top]
top_dipg = top_dipg[:top]
top_kcores = top_kcores[:top]


#------- 1st comparison (with simulate_spreading)
print('simulated spreading of top indegree is: ', simulate_spreading(G, top_indegree))
print('simulated spreading of top degree is: ', simulate_spreading(G, top_degree))
print('simulated spreading of top pagerank is: ', simulate_spreading(G, top_pg))
print('simulated spreading of top dir. indegree is: ', simulate_spreading(G, top_dipg))
print('simulated spreading of top kcores is: ', simulate_spreading(G, top_kcores))
print('simulated spreading of greedy sol is: ', simulate_spreading(G, greedy_set))

#------- 2nd comparison (with dni)
print('dni of top indegree is: ', dni(node_cascades, top_indegree))
print('dni of top degree is: ', dni(node_cascades, top_degree))
print('dni of top pagerank is: ', dni(node_cascades, top_pg))
print('dni of top dir. indegree is: ', dni(node_cascades, top_dipg))
print('dni of top kcores is: ', dni(node_cascades, top_kcores))
print('dni of greedy sol is: ', dni(node_cascades, greedy_set))




