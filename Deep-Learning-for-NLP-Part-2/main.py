import os
os.chdir("C:/Users/utilisateur/Desktop/LAST_YEAR/AlteGrad/ALTEGRAD/LAB5")

import sys
import json
import operator
import numpy as np

from sklearn.decomposition import PCA

from gensim.models import KeyedVectors

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras import backend as K
from keras.models import Model
from keras.backend.tensorflow_backend import _to_tensor
from keras.layers import Input, Embedding, Dropout, Bidirectional, GRU, TimeDistributed, Dense

path_root = './'
path_to_data = path_root + 'data/'

sys.path.insert(0, path_root)

from AttentionWithContext import AttentionWithContext

def bidir_gru(my_seq,n_units):
    '''
    just a convenient wrapper for bidirectional RNN with GRU units
    '''
    ### fill the gap ### # add a default GRU layer (https://keras.io/layers/recurrent/). You need to specify only the 'units' and 'return_sequences' arguments
    return Bidirectional(GRU(n_units, return_sequences=True)
                         ,merge_mode='concat', weights=None)(my_seq)
 
# = = = = = parameters = = = = =

n_units = 50
drop_rate = 0.5 
mfw_idx = 2 # index of the most frequent words in the dictionary. 
            # 0 is for the special padding token
            # 1 is for the special out-of-vocabulary token

padding_idx = 0
oov_idx = 1
batch_size = 32
nb_epochs = 6
my_optimizer = 'adam'
my_patience = 2 # for early stopping strategy

# = = = = = data loading = = = = =

my_docs_array_train = np.load(path_to_data + 'docs_train.npy')
my_docs_array_test = np.load(path_to_data + 'docs_test.npy')

my_labels_array_train = np.load(path_to_data + 'labels_train.npy')
my_labels_array_test = np.load(path_to_data + 'labels_test.npy')

# load dictionary of word indexes (sorted by decreasing frequency across the corpus)
with open(path_to_data + 'word_to_index.json', 'r') as my_file:
    word_to_index = json.load(my_file)

# invert mapping
index_to_word = dict((v,k) for k,v in word_to_index.items())

# = = = = = loading pretrained word vectors = = = = =

wvs = KeyedVectors.load(path_to_data + 'word_vectors.kv', mmap='r')

assert len(wvs.wv.vocab) == len(word_to_index) + 1 # vocab does not contain the OOV token

word_vecs = wvs.wv.syn0

pad_vec = np.random.normal(size=word_vecs.shape[1])

# add Gaussian vector on top of embedding matrix (padding vector)
word_vecs = np.insert(word_vecs,0,pad_vec,0)

print('embeddings created')

# reduce dimension with PCA (to reduce the number of parameters of the model)
my_pca = PCA(n_components=64)
embeddings_pca = my_pca.fit_transform(word_vecs)

print('embeddings compressed')

# = = = = = defining architecture = = = = =

# = = = sentence encoder

sent_ints = Input(shape=(my_docs_array_train.shape[2],)) # vec of ints of variable size

sent_wv = Embedding(input_dim=embeddings_pca.shape[0], # vocab size
                    output_dim=embeddings_pca.shape[1], # dimensionality of embedding space
                    weights=[embeddings_pca],
                    input_length=my_docs_array_train.shape[2],
                    trainable=True
                    )(sent_ints)

sent_wv_dr = Dropout(drop_rate)(sent_wv)

### fill the gap (3 gaps) ###
# use bidir_gru, AttentionWithContext with return_coefficients=True, and Dropout
sent_att_1 = bidir_gru(sent_wv_dr, n_units)
sent_att_2, sent_att_coeffs = AttentionWithContext(return_coefficients=True)(sent_att_1)
sent_att_vec_dr = Dropout(drop_rate)(sent_att_2)

sent_encoder = Model(sent_ints,sent_att_vec_dr)

# = = = document encoder

doc_ints = Input(shape=(my_docs_array_train.shape[1],my_docs_array_train.shape[2],))

### fill the gap (4 gaps) ###
# use TimeDistributed (https://keras.io/layers/wrappers/), bidir_gru, AttentionWithContext with return_coefficients=True, and Dropout
                  
doc_att_vec_dr_1 = TimeDistributed()(doc_ints)
doc_att_vec_dr_2 =  bidir_gru(doc_att_vec_dr_1, n_units)
doc_att_vec_dr_3,doc_att_coeffs = AttentionWithContext(return_coefficients=True)(doc_att_vec_dr_2)
doc_att_vec_dr = Dropout(drop_rate)(doc_att_vec_dr_3)

preds = Dense(units=1,
              activation='sigmoid')(doc_att_vec_dr)

model = Model(doc_ints,preds)

model.compile(loss='binary_crossentropy',
              optimizer = my_optimizer,
              metrics = ['accuracy'])

print('model compiled')

# = = = = = training = = = = =

loading_pretrained = False

if not loading_pretrained:
    early_stopping = EarlyStopping(monitor='val_acc', # go through epochs as long as accuracy on validation set increases
                                   patience=my_patience,
                                   mode='max')
    
    # save model corresponding to best epoch
    checkpointer = ModelCheckpoint(filepath=path_to_data + 'my_model', 
                                   verbose=1, 
                                   save_best_only=True,
                                   save_weights_only=True)
    
    # 200s/epoch on CPU - reaches 84.38% accuracy in 2 epochs
    model.fit(my_docs_array_train, 
              my_labels_array_train,
              batch_size = batch_size,
              epochs = nb_epochs,
              validation_data = (my_docs_array_test, my_labels_array_test) ,### fill the gap ### specify validation data as tuple
              callbacks = [early_stopping,checkpointer])

else:
    model.load_weights(path_to_data + 'my_model')

# = = = = = extraction of attention coefficients = = = = =

# define intermediate models (alternative to K.functions)
### fill the gap (2 gaps) ###
# define a Model named 'get_word_att_coeffs' that extracts the attention coefficients over the words in a sentence
# define a Model named 'get_sent_attention_coeffs' that extracts the attention coefficients over the sentences in a document
# in each case, use the right inputs, and as outputs, the coefficients returned by the corresponding AttentionWithContext layer
get_word_att_coeffs = Model(sent_ints,sent_att_coeffs) 
get_sent_attention_coeffs = Model(doc_ints,doc_att_coeffs) 


my_review = my_docs_array_test[-1:,:,:] # select last review
# convert integer review to text
index_to_word[1] = 'OOV'
my_review_text = [[index_to_word[idx] for idx in sent if idx in index_to_word] for sent in my_review.tolist()[0]]

# = = = attention over sentences in the document

sent_coeffs = get_sent_attention_coeffs.predict(my_review)
sent_coeffs = sent_coeffs[0,:,:]

for elt in zip(sent_coeffs[:,0].tolist(),[' '.join(elt) for elt in my_review_text]):
    print(round(elt[0]*100,2),elt[1])

# = = = attention over words in each sentence

my_review_tensor = _to_tensor(my_review,dtype='float32') # a layer, unlike a model, requires a TensorFlow tensor as input

### fill the gap (one line) ###
# apply the 'get_word_att_coeffs' model over all the sentences in 'my_review_tensor', and store the results as 'word_coeffs'

word_coeffs = TimeDistributed(get_word_att_coeffs)(my_review_tensor)
word_coeffs = K.eval(word_coeffs) # shape = (1, 7, 30, 1): (batch size, nb of sents in doc, nb of words per sent, coeff)

word_coeffs = word_coeffs[0,:,:,0] # shape = (7, 30) (coeff for each word in each sentence)

word_coeffs = sent_coeffs * word_coeffs # re-weigh according to sentence importance

word_coeffs = np.round((word_coeffs*100).astype(np.float64),2)

word_coeffs_list = word_coeffs.tolist()

# match text and coefficients
text_word_coeffs = [list(zip(words,word_coeffs_list[idx][:len(words)])) for idx,words in enumerate(my_review_text)]

for sent in text_word_coeffs:
    [print(elt) for elt in sent]
    print('= = = =')

# sort words by importance within each sentence
text_word_coeffs_sorted = [sorted(elt,key=operator.itemgetter(1),reverse=True) for elt in text_word_coeffs]

for sent in text_word_coeffs_sorted:
    [print(elt) for elt in sent]
    print('= = = =')
