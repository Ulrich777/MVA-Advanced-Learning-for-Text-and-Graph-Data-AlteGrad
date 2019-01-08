"""
Graph Mining - ALTEGRAD - Dec 2018
"""

import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
from grakel.kernels import ShortestPath, PyramidMatch, RandomWalk, VertexHistogram, WeisfeilerLehman
from grakel import graph_from_networkx
from grakel.datasets import fetch_dataset
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


############## Question 1
# Generate simple dataset

Gs = list()
y = list()

##################
# your code here #
for i in range(100):
    Gs.append(nx.path_graph(i+3))
    y.append(1)
    Gs.append(nx.cycle_graph(i+3))
    y.append(0)
    
##################



############## Question 2
# Classify the synthetic graphs using graph kernels

# Split dataset into a training and a test set
# hint: use the train_test_split function of scikit-learn

##################
# your code here #
G_train , G_test , y_train , y_test = train_test_split(Gs, y, test_size= 0.1)
##################

# Transform NetworkX graphs to objects that can be processed by GraKeL
G_train = list(graph_from_networkx(G_train))
G_test = list(graph_from_networkx(G_test))


# Use the shortest path kernel to generate the two kernel matrices ("K_train" and "K_test")
# hint: the graphs do not contain node labels. Set the with_labels argument of the the shortest path kernel to False

##################
# your code here #
gk = ShortestPath(with_labels=False)
K_train = gk.fit_transform(G_train)
K_test = gk.transform(G_test)
##################


clf = SVC(kernel='precomputed', C=1) # Initialize SVM
clf.fit(K_train, y_train) # Train SVM
y_pred = clf.predict(K_test) # Predict

# Compute the classification accuracy
# hint: use the accuracy_score function of scikit-learn


##################
# your code here #
print(accuracy_score(y_test, y_pred))
##################


# Use the random walk kernel and the pyramid match graph kernel to perform classification

##################
# your code here #
gk1 = RandomWalk()
K_train1 = gk1.fit_transform(G_train)
K_test1 = gk1.transform(G_test)

clf1 = SVC(kernel='precomputed', C=1) # Initialize SVM
clf1.fit(K_train, y_train) # Train SVM
y_pred1 = clf1.predict(K_test) # Predict

print(accuracy_score(y_test, y_pred1))
##################


############## Question 3
# Classify the graphs of a real-world dataset using graph kernels

# Load the MUTAG dataset
# hint: use the fetch_dataset function of GraKeL

##################
# your code here #
mutag = fetch_dataset('MUTAG', verbose=False)
G, y = mutag.data, mutag.target
##################


# Split dataset into a training and a test set
# hint: use the train_test_split function of scikit-learn

##################
# your code here #
G_train , G_test , y_train , y_test = train_test_split(G, y, test_size= 0.1)
##################


# Perform graph classification using different kernels and evaluate performance

##################
# your code here #
kernels = [VertexHistogram, ShortestPath, PyramidMatch, WeisfeilerLehman]
names = ['Vertex Histogram', 'Shortest Path Kernel', 
         'Pyramid Match Graph Kernel', 'Weisfeiler-Lehman Subtree Kernel']

for kernel, name in zip(kernels, names):
    print(name+'......')
    if name in ['Shortest Path Kernel', 'Pyramid Match Graph Kernel']:
        gk_ = kernel(with_labels=True)
    elif name == 'Weisfeiler-Lehman Subtree Kernel':
        gk_ = kernel(base_kernel=VertexHistogram)
    else:
        gk_ = kernel()
    K_train_ = gk_.fit_transform(G_train)
    K_test_ = gk_.transform(G_test)

    clf_ = SVC(kernel='precomputed', C=1) # Initialize SVM
    clf_.fit(K_train_, y_train) # Train SVM
    y_pred_ = clf_.predict(K_test_) # Predict

    print(accuracy_score(y_test, y_pred_))
##################