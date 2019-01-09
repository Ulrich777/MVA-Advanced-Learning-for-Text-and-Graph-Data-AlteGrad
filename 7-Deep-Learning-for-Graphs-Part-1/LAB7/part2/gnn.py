"""
Deep Learning on Graphs - ALTEGRAD - Jan 2019
"""
import os
os.chdir("C:/Users/utilisateur/Desktop/LAST_YEAR/AlteGrad/ALTEGRAD/LAB7/part2")

import numpy as np
import time
import matplotlib.pyplot as plt
from keras.layers import Input
from keras.layers import Dropout,Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.metrics import accuracy_score, log_loss
from sklearn.manifold import TSNE

from utils import load_data
from layers import MessagePassing


# Read data
features, adj, y = load_data()
n = adj.shape[0] # Number of nodes

############## Question 1
# Set each component of the main diagonal of the adjacency matrix to 1


##################
# your code here #
for i in range(n):
    adj[i,i] = 1.
##################

# Normalize the emerging matrix such that each row sums to 1

##################
# your code here #
##################
D = np.sqrt(1/np.sum(adj, axis=1))
d = np.diag(D.A1)

adj = np.dot(d, np.dot(adj,d))


# Yields indices to split data into training, validation and test sets
idx = np.random.permutation(n)
idx_train = idx[:int(0.6*n)]
idx_val = idx[int(0.6*n):int(0.8*n)]
idx_test = idx[int(0.8*n):]

# Produces a mask to identify training examples
train_mask = np.zeros(n)
train_mask[idx_train] = 1
train_mask = np.array(train_mask, dtype=np.bool)

# Define model architecture
graph = [features, adj]
G = [Input(shape=(None, None), batch_shape=(None, None), sparse=False)]

X_in = Input(shape=(features.shape[1],))


############## Question 2
# Use the functional API of Keras to define the layers of the model
# hint: The message passing layer is MessagePassing(units, activation)([H]+G)

##################
# your code here #
##################
X_in_dr = Dropout(0.5)(X_in)
H_prev = MessagePassing(32, activation='relu')([X_in]+G)
H_prev_dr = Dropout(0.5)(H_prev)
H = MessagePassing(16, activation = 'relu')([H_prev_dr]+G)
Y = Dense(units=7,
              activation='softmax')(H)


# Create the model and compile it (Use the name "model" for the model)

##################
# your code here #
##################
model = Model([X_in] + G, Y)
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01))

# Helper variables for main training loop
epochs = 100
wait = 0
best_val_loss = 99999
PATIENCE = 10

# Fit
for epoch in range(1, epochs+1):

    # Log wall-clock time
    t = time.time()

    # Single training iteration (we mask nodes without labels for loss calculation)
    model.fit(graph, y, sample_weight=train_mask,
              batch_size=adj.shape[0], epochs=1, shuffle=False, verbose=0)

    # Predict on full dataset
    preds = model.predict(graph, batch_size=adj.shape[0])

    # Train / validation scores
    train_loss = log_loss(y[idx_train,:], preds[idx_train,:])
    train_acc = accuracy_score(np.argmax(preds[idx_train,:], 1), np.argmax(y[idx_train,:], 1))
    val_loss = log_loss(y[idx_val,:], preds[idx_val,:])
    val_acc = accuracy_score(np.argmax(preds[idx_val,:], 1), np.argmax(y[idx_val,:], 1))

    print("Epoch: {:04d}".format(epoch),
          "train_loss= {:.4f}".format(train_loss),
          "train_acc= {:.4f}".format(train_acc),
          "val_loss= {:.4f}".format(val_loss),
          "val_acc= {:.4f}".format(val_acc),
          "time= {:.4f}".format(time.time() - t))

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        wait = 0
    else:
        if wait >= PATIENCE:
            print('Epoch {}: early stopping'.format(epoch))
            break
        wait += 1

# Testing
test_loss = log_loss(y[idx_test,:], preds[idx_test,:])
test_acc = accuracy_score(np.argmax(preds[idx_test,:], 1), np.argmax(y[idx_test,:], 1))
print("Test set results:",
      "loss= {:.4f}".format(test_loss),
      "accuracy= {:.4f}".format(test_acc))



############## Question 3
# Create a new model that uses the layers defined above, but its output is the output of the second message passing layer. Use the new model to obtain representations for the nodes of the test set

##################
# your code here #
model_ = Model([X_in] + G, H)
preds_ = model_.predict(graph, batch_size=adj.shape[0])
node_emb_test = preds_[idx_test,:]

my_tsne = TSNE(n_components=2,perplexity=10)
node_emb_test =  my_tsne.fit_transform(node_emb_test)



# Project the emerging representations to two dimensions using t-SNE. Store the new representations in the variable "node_emb_test" 

##################
# your code here #
##################


labels = np.argmax(y[idx_test,:], axis=1)
unique_labels = np.unique(labels)

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

fig, ax = plt.subplots()
for i in range(unique_labels.size):
    idxs = [j for j in range(labels.size) if labels[j]==unique_labels[i]]
    ax.scatter(node_emb_test[idxs,0], 
               node_emb_test[idxs,1], 
               c=colors[i],
               label=i,
               alpha=0.7,
               s=10)

ax.legend(scatterpoints=1)
fig.suptitle('T-SNE Visualization of the nodes of the test set',fontsize=12)
fig.set_size_inches(6,4)
plt.show()