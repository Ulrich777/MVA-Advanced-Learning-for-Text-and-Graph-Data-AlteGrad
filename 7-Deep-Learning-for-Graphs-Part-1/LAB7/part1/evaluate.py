"""
Deep Learning on Graphs - ALTEGRAD - Jan 2019
"""

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
import csv
from sklearn.metrics import f1_score
from scipy.io import loadmat
from sklearn.utils import shuffle as skshuffle
import numpy as np
from collections import defaultdict
import sys

############## Question 3
# Evaluate node embedding algorithms

# Specify the file where the embeddings are stored and their dimensionality

embeddings_file = 'embeddings/deepwalk_embeddings'
                  # your code here #
embeddings_dim =  128


class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        probs = super(TopKRanker, self).predict_proba(X)
        y_pred = np.zeros(probs.shape)
        for i in range(len(top_k_list)):
            probs_ = probs[i,:].argsort()[::-1]
            for j in range(int(top_k_list[i][0])):
                y_pred[i,probs_[j]] = 1
        return y_pred

# 0. Files
matfile = "data/Homo_sapiens.mat"

# 1. Load labels
mat = loadmat(matfile)
labels_matrix = mat['group']

# 2. Load Embeddings
features_matrix = np.zeros((labels_matrix.shape[0],embeddings_dim))

with open(embeddings_file,'r') as f:
    reader=csv.reader(f,delimiter=' ')
    for row in reader:
        features_matrix[int(row[0]),:] = np.array(row[1:])


# 3. Shuffle, to create train/test groups
shuffles = []
number_shuffles = 5
for x in range(number_shuffles):
    shuffles.append(skshuffle(features_matrix, labels_matrix))

# 4. to score each train/test group
all_results = defaultdict(list)

training_percents = [0.9]
for train_percent in training_percents:
    for shuf in shuffles:

        X, y = shuf

        training_size = int(train_percent * X.shape[0])

        X_train = X[:training_size, :]
        y_train = y[:training_size]

        X_test = X[training_size:, :]
        y_test = y[training_size:]

        clf = TopKRanker(LogisticRegression(solver="lbfgs"))
        clf.fit(X_train, y_train)

        # find out how many labels should be predicted
        top_k_list = np.sum(y_test, axis=1).tolist()
        y_pred = clf.predict(X_test, top_k_list)

        results = {}
        averages = ["micro", "macro"]
        for average in averages:
            results[average] = f1_score(y_test,  y_pred, average=average)

        all_results[train_percent].append(results)

print('Results, using embeddings of dimensionality', X.shape[1])
print('-------------------')
for train_percent in sorted(all_results.keys()):
    print('Train percent:', train_percent)
    for x in all_results[train_percent]:
        print(x)
    print('-------------------')
