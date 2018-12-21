import os
os.chdir("C:/Users/utilisateur/Desktop/LAST_YEAR/AlteGrad/ALTEGRAD/LAB4")

import csv
import json
import numpy as np

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from gensim.models.word2vec import Word2Vec

from keras.models import Model, load_model
from keras import backend as K
from keras.layers import Input, Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Concatenate, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint

path_to_data = './data/'
path_to_word2vec = "C:/Users/utilisateur/Desktop/LAST_YEAR/AlteGrad/ALTEGRAD/LAB3/"


# = = = = = functions = = = = =

def cnn_branch(n_filters,k_size,d_rate,my_input):
    return Dropout(d_rate)(GlobalMaxPooling1D()(Conv1D(filters=n_filters,
                                                       kernel_size=k_size,
                                                       activation='relu')(my_input)))

# = = = = = parameters = = = = =

mfw_idx = 2 # index of the most frequent words in the dictionary. 
            # 0 is for the special padding token
            # 1 is for the special out-of-vocabulary token

padding_idx = 0
oov_idx = 1

initial_word_vector_dim = 300 # dimensionality of the words vectors (must match that of the pretrained!)
max_size = 200 # max allowed size of a document
nb_branches = 2
nb_filters = 150
filter_sizes = [3,4]
drop_rate = 0.3 # amount of dropout regularization
batch_size = 32
nb_epochs = 6
my_optimizer = 'adam'
my_patience = 2 # for early stopping strategy

# = = = = = data loading = = = = =

# load dictionary of word indexes (sorted by decreasing frequency across the corpus)
with open(path_to_data + 'word_to_index.json', 'r') as my_file:
    word_to_index = json.load(my_file)

# invert mapping
index_to_word = dict((v,k) for k,v in word_to_index.items())

with open(path_to_data + 'training.csv', 'r') as my_file:
    reader = csv.reader(my_file, delimiter=',')
    x_train = list(reader)

with open(path_to_data + 'test.csv', 'r') as my_file:
    reader = csv.reader(my_file, delimiter=',')
    x_test = list(reader)

with open(path_to_data + 'training_labels.txt', 'r') as my_file:
    y_train = my_file.read().splitlines()

with open(path_to_data + 'test_labels.txt', 'r') as my_file:
    y_test = my_file.read().splitlines()

# turn lists of strings into lists of integers
x_train = [[int(elt) for elt in sublist] for sublist in x_train]
x_test = [[int(elt) for elt in sublist] for sublist in x_test]  

y_train = [int(elt) for elt in y_train]
y_test = [int(elt) for elt in y_test]

# = = some sanity checking = =

inspect_index = True
if inspect_index:
    
    print('index of "the":',word_to_index['the']) # most frequent word
    print('index of "movie":',word_to_index['movie']) # very frequent word
    print('index of "elephant":',word_to_index['elephant']) # less frequent word
        
    # reconstruct first review
    rev = x_train[0]
    print (' '.join([index_to_word[elt] if elt in index_to_word else 'OOV' for elt in rev]))
    
    # compare it with the original review: https://www.imdb.com/review/rw2219371/?ref_=tt_urv

print('data loaded')

# = = = = = truncation and padding = = = = =

# truncate reviews longer than 'max_size'
x_train = [rev[:max_size] for rev in x_train]
x_test = [rev[:max_size] for rev in x_test]

### fill gap ###
# pad reviews shorter than 'max_size' with padding_idx - you may use list comprehensions


x_train = [d + [padding_idx]*(max_size - len(d)) for d in x_train]
x_test = [d + [padding_idx]*(max_size - len(d)) for d in x_test]

# all reviews should now be of size 'max_size'
assert max_size == list(set([len(rev) for rev in x_train]))[0] and max_size == list(set([len(rev) for rev in x_test]))[0]

print('truncation and padding done')

# = = = = = loading pretrained word vectors = = = = =

word_vectors = Word2Vec(size=initial_word_vector_dim, min_count=1) # initialize word vectors

# create entries for the words in our vocabulary
word_vectors.build_vocab([[elt] for elt in list(index_to_word.values())]) # build_vocab takes a list of list as input

# fill entries with the pre-trained word vectors
word_vectors.intersect_word2vec_format(path_to_word2vec + 'GoogleNews-vectors-negative300.bin.gz', binary=True)

print('pre-trained word vectors loaded')

# get numpy array of embeddings  
embeddings = word_vectors.wv.syn0

# add zero vector (for padding special token)
pad_vec = np.zeros((1,initial_word_vector_dim))
embeddings = np.insert(embeddings,0,pad_vec,0)

# add Gaussian initialized vector (for OOV special token)
oov_vec = np.random.normal(size=initial_word_vector_dim) 
embeddings = np.insert(embeddings,0,oov_vec,0)

print('embeddings created')

# reduce dimension with PCA (to reduce the number of parameters of the model)
my_pca = PCA(n_components=64)
embeddings_pca = my_pca.fit_transform(embeddings)

print('embeddings compressed')

# = = = = = defining architecture = = = = =

# see guide to Keras' functional API: https://keras.io/getting-started/functional-api-guide/
# core layers: https://keras.io/layers/core/
# conv layers: https://keras.io/layers/convolutional/
# pooling layers: https://keras.io/layers/pooling/

doc_ints = Input(shape=(None,))

### fill the gap ###
# add an Embedding layer with input_dim, output_dim, weights, input_length, and trainable arguments
doc_wv = Embedding(max(word_to_index.values())+1, 64, input_length=max_size)(doc_ints)
doc_wv_dr = Dropout(drop_rate)(doc_wv)

branch_outputs = []
for idx in range(nb_branches):
    ### fill the gap ###
    # use the cnn_branch function
    conv = cnn_branch(nb_filters,filter_sizes[idx],drop_rate, doc_wv_dr)
    branch_outputs.append(conv)

concat = Concatenate()(branch_outputs)

### fill the gap ###
# add a dense layer with the proper number of units and the proper activation function
concat = Dense(50, activation="relu")(concat)
preds = Dense(1, activation="sigmoid")(concat)
model = Model(doc_ints,preds)

model.compile(loss='binary_crossentropy',
              optimizer = my_optimizer,
              metrics = ['accuracy'])

print('model compiled')

model.summary()

print('total number of model parameters:',model.count_params())

# = = = = = visualizing doc embeddings (before training) = = = = =

# you can access the layers of the model with model.layers
# then the input/output shape of each layer with, e.g., model.layers[0].input_shape or model.layers[0].output_shape

# extract output of the final embedding layer (before the softmax)
# in test mode, we should set the 'learning_phase' flag to 0 (e.g., we don't want to use dropout)
get_doc_embedding = K.function([model.layers[0].input,K.learning_phase()],
                               [model.layers[9].output])

n_plot = 1000
print('plotting embeddings of first',n_plot,'documents')

doc_emb = get_doc_embedding([np.array(x_test[:n_plot]),0])[0]

my_pca = PCA(n_components=10)
my_tsne = TSNE(n_components=2,perplexity=10) #https://lvdmaaten.github.io/tsne/
doc_emb_pca = my_pca.fit_transform(doc_emb) 
doc_emb_tsne = my_tsne.fit_transform(doc_emb_pca)

labels_plt = y_test[:n_plot]
my_colors = ['blue','red']

fig, ax = plt.subplots()

for label in list(set(labels_plt)):
    idxs = [idx for idx,elt in enumerate(labels_plt) if elt==label]
    ax.scatter(doc_emb_tsne[idxs,0], 
               doc_emb_tsne[idxs,1], 
               c = my_colors[label],
               label=str(label),
               alpha=0.7,
               s=40)

ax.legend(scatterpoints=1)
fig.suptitle('t-SNE visualization of CNN-based doc embeddings \n (first 1000 docs from test set)',fontsize=15)
fig.set_size_inches(11,7)
fig.show()

# = = = = = training = = = = =

# warning: by default on CPU, will use all cores
# reaches ~86% val accuracy in 4 epochs (8 secs/epoch on Titan GPU)

early_stopping = EarlyStopping(monitor='val_acc', # go through epochs as long as accuracy on validation set increases
                               patience=my_patience,
                               mode='max')

# save model corresponding to best epoch
checkpointer = ModelCheckpoint(filepath=path_to_data + 'my_model', 
                               verbose=1, 
                               save_best_only=True)

### fill the gap ###
# call the fit() method on model https://keras.io/models/model/ with the proper arguments
# convert x_train and x_test to numpy arrays
x_train, x_test = np.array(x_train), np.array(x_test)
model.fit(x_train, y_train, batch_size=batch_size, epochs=4,
          validation_data=(x_test, y_test), verbose=2, callbacks=[early_stopping, checkpointer])
# to load pre-trained model, if necessary: model = load_model(path_to_data + 'model')

# = = = = = visualizing doc embeddings (after training) = = = = =

# perform the same steps as before training and observe the changes
get_doc_embedding = K.function([model.layers[0].input,K.learning_phase()],
                               [model.layers[9].output])
n_plot = 1000
print('plotting embeddings of first',n_plot,'documents')

doc_emb = get_doc_embedding([np.array(x_test[:n_plot]),0])[0]

my_pca = PCA(n_components=10)
my_tsne = TSNE(n_components=2,perplexity=10) #https://lvdmaaten.github.io/tsne/
doc_emb_pca = my_pca.fit_transform(doc_emb) 
doc_emb_tsne = my_tsne.fit_transform(doc_emb_pca)

labels_plt = y_test[:n_plot]
my_colors = ['blue','red']

fig, ax = plt.subplots()

for label in list(set(labels_plt)):
    idxs = [idx for idx,elt in enumerate(labels_plt) if elt==label]
    ax.scatter(doc_emb_tsne[idxs,0], 
               doc_emb_tsne[idxs,1], 
               c = my_colors[label],
               label=str(label),
               alpha=0.7,
               s=40)

ax.legend(scatterpoints=1)
fig.suptitle('t-SNE visualization of CNN-based doc embeddings \n (first 1000 docs from test set)',fontsize=15)
fig.set_size_inches(11,7)
fig.show()


# = = = = = predictive text regions for the first branch = = = = =

### fill the gap ###
# create a K.function named 'get_region_embedding' that extracts the region embeddings
get_region_embedding = K.function([model.layers[0].input,K.learning_phase()],
                               [model.layers[3].output])

### fill the gap ###
# create a K.function named 'get_sigmoid' that extracts the final prediction
get_sigmoid = K.function([model.layers[0].input,K.learning_phase()],
                               [model.layers[11].output])

my_review = x_test[10]

tokens = ['OOV' if elt==1 else index_to_word[elt] for elt in my_review if elt!=0]

# extract regions (sliding window over text)
regions = []
regions.append(' '.join(tokens[:filter_sizes[0]]))
for i in range(filter_sizes[0], len(tokens)):
    regions.append(' '.join(tokens[(i-filter_sizes[0]+1):(i+1)]))

my_review = np.array([my_review])

reg_emb = get_region_embedding([my_review,0])[0]

prediction = get_sigmoid([my_review,0])[0]

### fill the gap ###
# compute the norm of each row of reg_emb[0,:,:] using the np.linalg.norm function with the proper axis argument
# store the results as 'norms'
norms = np.linalg.norm(reg_emb[0,:,:], axis=1)
norms = norms[1:len(regions)]

print([list(zip(regions,norms))[idx] for idx in np.argsort(-norms).tolist()])

# = = = = = saliency map = = = = =

input_tensors = [model.input, K.learning_phase()]
saliency_input = model.layers[3].input # before convolution
saliency_output = model.layers[10].output # class score

gradients = model.optimizer.get_gradients(saliency_output,saliency_input)
compute_gradients = K.function(inputs=input_tensors,outputs=gradients)

### fill the gap ###
# save the result of compute_gradients as an object named 'matrix'
matrix = compute_gradients([my_review,0])
matrix = matrix[0][0,:,:] # should be of shape (200,64)

to_plot = np.absolute(matrix[:len(tokens),:])

fig, ax = plt.subplots()
heatmap = ax.imshow(to_plot, cmap=plt.cm.Blues, interpolation='nearest',aspect='auto')
ax.set_yticks(np.arange(len(tokens)))
ax.set_yticklabels(tokens)
ax.tick_params(axis='y', which='major', labelsize=32*10/len(tokens))
fig.colorbar(heatmap)
fig.set_size_inches(11,7)
fig.savefig(path_to_data + 'saliency_map.png',bbox_inches='tight')
fig.show()
