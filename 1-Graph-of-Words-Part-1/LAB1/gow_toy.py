import string
import os
os.chdir("C:/Users/utilisateur/Desktop/AlteGrad/ALTEGRAD/for_moodle 2/code")



from nltk.corpus import stopwords
import networkx as nx
#import matplotlib.pyplot as plt


# import custom functions
from library import clean_text_simple, terms_to_graph, unweighted_k_core

stpwds = stopwords.words('english')
punct = string.punctuation.replace('-', '')

my_doc = '''A method for solution of systems of linear algebraic equations 
with m-dimensional lambda matrices. A system of linear algebraic 
equations with m-dimensional lambda matrices is considered. 
The proposed method of searching for the solution of this system 
lies in reducing it to a numerical system of a special kind.'''

my_doc = my_doc.replace('\n', '')

# pre-process document
my_tokens = clean_text_simple(my_doc,my_stopwords=stpwds,punct=punct)
                              
g = terms_to_graph(my_tokens, w=4)

#nx.draw(g, with_labels=True)
#plt.show()
    
# number of edges
print(len(list(g.edges)))

# the number of nodes should be equal to the number of unique terms
len(list(g.nodes)) == len(set(my_tokens))

edge_weights = []
for (source,target,weight) in g.edges(data='weight'):
    edge_weights.append([source,target,weight])



print(edge_weights)

for w in range(2,11):
    g = terms_to_graph(my_tokens, w)
    # print density of g
    print(nx.density(g))
    
# decompose g
core_numbers = unweighted_k_core(g)  # fill the gap #
print(core_numbers)

# compare with igraph method

cn = nx.core_number(nx.Graph(g))



# retain main core as keywords
max_c_n = max(core_numbers.values())
keywords = [kwd for kwd, c_n in core_numbers.items() if c_n == max_c_n]
print(keywords)












