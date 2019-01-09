"""
Deep Learning on Graphs - ALTEGRAD - Jan 2019
"""
import os
os.chdir("C:/Users/utilisateur/Desktop/LAST_YEAR/AlteGrad/ALTEGRAD/LAB7/part1")



import numpy as np
import networkx as nx
from random import randint
from scipy.io import loadmat
from gensim.models import Word2Vec


############## Question 2
# Implement the deepwalk algorithm

def random_walk(G, node, walk_length):
	# Starts from vertex "node" and performs a random walk of length "walk length". Returns a list of the visited vertices
	
	##################
	# your code here #
	##################
    walk = [node]
    
    for i in range(walk_length-1):
        new = np.random.choice(list(G.neighbors(walk[-1])))
        walk.append(new)

    walk = [str(node) for node in walk]
    return walk


def generate_walks(graph, num_walks, walk_length):
	# Runs "num_walks" random walks from each node, and returns a list of all random walk
	
	##################
	# your code here #
	##################
    walks = []
    for node in graph.nodes():
        for i in range(num_walks):
            walk = random_walk(graph, node, walk_length)
            walks.append(walk)

    return walks


def learn_embeddings_and_write_to_disk(graph, walks, window_size, d):
	model = Word2Vec(walks, size=d, window=window_size, min_count=0, sg=1, workers=8, iter=2)
	fout = open('embeddings/deepwalk_embeddings', 'w', encoding="UTF-8")
	for i in range(G.number_of_nodes()):
		e = model.wv[str(i)]
		e = ' '.join(map(lambda x: str(x), e))
		fout.write('%s %s\n' % (i, e))


if __name__ == '__main__':
	d = {}
	loadmat('data/Homo_sapiens.mat', mdict=d)
	A = d['network']
	G = nx.from_scipy_sparse_matrix(A)
	walks = generate_walks(G, 7, 40)
	learn_embeddings_and_write_to_disk(G, walks, 10, 128)