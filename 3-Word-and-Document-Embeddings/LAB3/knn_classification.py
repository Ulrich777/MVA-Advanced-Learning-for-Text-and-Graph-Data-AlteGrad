import os
os.chdir("C:/Users/utilisateur/Desktop/LAST_YEAR/AlteGrad/ALTEGRAD/LAB3")


import re
import random
import string
import time
import operator
import numpy as np
import multiprocessing
from functools import partial
from multiprocessing import Pool
from collections import Counter
from gensim.models.word2vec import Word2Vec
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity as cosine
from sklearn.feature_extraction.text import TfidfVectorizer

random.seed(111417)

# remove dashes and apostrophes from punctuation marks 
punct = string.punctuation.replace('-', '').replace("'",'')
# regex to match intra-word dashes and intra-word apostrophes
my_regex = re.compile(r"(\b[-']\b)|[\W_]")

path_root = "C:/Users/utilisateur/Desktop/LAST_YEAR/AlteGrad/ALTEGRAD/LAB3/"

path_to_data = path_root + 'data/'
path_to_documents = path_root + 'data/documents/'
path_to_google_news = path_root

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]

# returns the vector of a word
def my_vector_getter(word, wv):
    try:
        # we use reshape because cosine similarity in sklearn now works only for multidimensional arrays
        word_array = wv.wv[word].reshape(1,-1)
        return (word_array)
    except KeyError:
        print('word: <', word, '> not in vocabulary!')
    
# performs basic pre-processing
def clean_string(string, punct=punct, my_regex=my_regex, to_lower=False):
    if to_lower:
        string = string.lower()
    # remove formatting
    str = re.sub('\s+', ' ', string)
     # remove punctuation
    str = ''.join(l for l in str if l not in punct)
    # remove dashes that are not intra-word
    str = my_regex.sub(lambda x: (x.group(1) if x.group(1) else ' '), str)
    # strip extra white space
    str = re.sub(' +',' ',str)
    # strip leading and trailing white space
    str = str.strip()
    return str

def to_parallelize(doc,collection,w2v):
    to_return = []
    for doc_train in collection:
        ### fill gap ### append the Word Mover's Distance between doc and doc_train
        to_return.append(w2v.wv.wmdistance(document1=doc, document2=doc_train))
    return to_return

t = time.time()

with open(path_to_data + 'smart_stopwords.txt', 'r') as my_file: 
    stpwds = my_file.read().splitlines()

doc_names = os.listdir(path_to_documents)
doc_names.sort(key=natural_keys)
docs = []
for idx,name in enumerate(doc_names):
    with open(path_to_documents + name,'r') as my_file:
        docs.append(my_file.read())
    if idx % round(len(doc_names)/10) == 0:
        print(idx)

with open(path_to_data + 'labels.txt', 'r') as my_file: 
    labels = my_file.read().splitlines()

labels = np.array([int(item) for item in labels])

print('documents, labels and stopwords loaded in', round(time.time() - t,2), 'second(s)')

shuffled_idxs = random.sample(range(len(docs)), len(docs)) # sample w/o replct
docs = [docs[idx] for idx in shuffled_idxs]
labels = [labels[idx] for idx in shuffled_idxs]

print('documents and labels shuffled')

t = time.time()

cleaned_docs = []
for idx, doc in enumerate(docs):
    # clean
    doc = clean_string(doc, punct, my_regex, to_lower=True)
    # tokenize (split based on whitespace)
    tokens = doc.split(' ')
    # remove stopwords
    tokens = [token for token in tokens if token not in stpwds]
    # remove digits
    tokens = [''.join([elt for elt in token if not elt.isdigit()]) for token in tokens]
    # remove tokens shorter than 3 characters in size
    tokens = [token for token in tokens if len(token)>2]
    # remove tokens exceeding 25 characters in size
    tokens = [token for token in tokens if len(token)<=25]
    cleaned_docs.append(tokens)
    if idx % round(len(docs)/10) == 0:
        print(idx)

print('documents cleaned in', round(time.time() - t,2), 'second(s)')

# create empty word vectors for the words in vocabulary 
# we set size=300 to match dim of GNews word vectors
my_q = 300
mcount = 5
w2v = Word2Vec(size=my_q, min_count=mcount)

w2v.build_vocab(cleaned_docs)

## fill the gap ## w2v.wv.vocab returns a dictionary
vocab = w2v.wv.vocab
all_tokens = [token for sublist in cleaned_docs for token in sublist]
t_counts = dict(Counter(all_tokens))
assert len(vocab) == len([token for token,count in t_counts.items() if count>=mcount])

t = time.time()

w2v.intersect_word2vec_format(path_to_google_news + 'GoogleNews-vectors-negative300.bin.gz', binary=True)

print('word vectors loaded in', round(time.time() - t,2), 'second(s)')

# NOTE: in-vocab words without an entry in the Google News file are not removed from the vocabulary
# instead, their vectors are silently initialized to random values
# we can detect those vectors via their norms which approach zero
norms = [np.linalg.norm(w2v[word]) for word in vocab]
idxs_zero_norms = [idx for idx,norm in enumerate(norms) if norm<=0.05]
## fill the gap ## get the words with close to zero norms
no_entry_words = [word for idx, word in enumerate(vocab) if idx in idxs_zero_norms]

# remove no-entry words and infrequent words
no_entry_words = set(no_entry_words)
for idx,doc in enumerate(cleaned_docs):
    cleaned_docs[idx] = [token for token in doc if token not in no_entry_words and t_counts[token]>=mcount]
    if idx % round(len(docs)/10) == 0:
        print(idx)

# retain only 'max_size' first words of each doc to speed-up computation of WMD
max_size = 100

cleaned_docs = [elt[:max_size] for elt in cleaned_docs]

print('documents truncated to', max_size, 'word(s)')

# compute centroids of documents
t = time.time()

centroids = np.empty(shape=(len(cleaned_docs),my_q))
for idx,doc in enumerate(cleaned_docs):
    ## fill the gap ## compute the centroid by using mean and concatenate
    embeddings = np.concatenate([my_vector_getter(token,w2v) for token in doc])
    centroid = np.mean(embeddings, axis=0).reshape(1,-1)
    centroids[idx,:] = centroid
    if idx % round(len(docs)/10) == 0:
        print(idx)

print('centroids computed in', round(time.time() - t,2), 'second(s)')

# use the first n_train docs as training set and last n_test docs as test set
# compute distance between each element in the test set and each element in the training set

n_train = 100
n_test = 50

print('using', n_train, 'documents as examples')
print('using', n_test, 'documents for testing')

tfidf_vect = TfidfVectorizer(min_df=1, 
                             stop_words=None, 
                             lowercase=False, 
                             preprocessor=None)

# tfidf_vectorizer takes raw documents as input
doc_term_mtx = tfidf_vect.fit_transform([' '.join(elt) for elt in cleaned_docs[:n_train]])

t = time.time()

my_similarities = []
for idx,doc_test in enumerate(cleaned_docs[-n_test:]):
    # notice that we just transform
    doc_test_vect = tfidf_vect.transform([' '.join(doc_test)])
    sims = cosine(doc_term_mtx, Y=doc_test_vect, dense_output=True)
    my_similarities.append(sims[:,0])
    if idx % round(n_test/10) == 0:
        print(idx)

print('TFIDF cosine similarities computed in', round(time.time() - t,2), 'second(s)')

t = time.time()

my_centroid_similarities = []
for idx in range(n_test):
    sims = cosine(centroids[:n_train,:], 
                  Y=centroids[centroids.shape[0]-(idx+1),:].reshape(1, -1), 
                  dense_output=True)
    my_centroid_similarities.append(sims[:,0])
    if idx % round(n_test/10) == 0:
        print(idx)

print('centroid-based cosine similarities computed in', round(time.time() - t,2), 'second(s)')

t = time.time()

#to_parallelize_partial = partial(to_parallelize,
#                                 collection=cleaned_docs[:n_train],
#                                w2v=w2v)
 
#n_jobs = multiprocessing.cpu_count()

#print('using', n_jobs, 'core(s)')
#pool = Pool(processes=n_jobs)
#my_distances = pool.map(to_parallelize_partial, cleaned_docs[-n_test:])
#pool.close()
#pool.join()

# uncomment the lines below if parallelization (lines 215-224) does not work
my_distances = []
for idx,doc_test in enumerate(cleaned_docs[-n_test:]):
    tmp = []
    for doc_train in cleaned_docs[:n_train]:
        tmp.append(w2v.wv.wmdistance(doc_test,doc_train))
    my_distances.append(tmp)
    if idx % round(n_test/10) == 0:
        print(idx)

print('WM distances computed in', round(time.time() - t,2), 'second(s)')

print('========== performance of WMD ==========')

for nn in [1,3,5,7,11,13,15,17,21,23]:
    
    preds_wmd = []
    for idx,dists in enumerate(my_distances):
        idxs_sorted = np.argsort(dists).tolist() # by default, indexes of the elements sorted by increasing order!
        # get labels of 'nn' nearest neighbors
        labels_nn = [labels[:n_train][elt] for elt in idxs_sorted[:nn]] # the less distant elements are the closest
        # select most frequent label as prediction
        counts = dict(Counter(labels_nn))
        max_counts = max(list(counts.values()))
        pred = [k for k,v in counts.items() if v==max_counts][0]
        preds_wmd.append(pred)
    
    # compare predictions to true labels
    
    print('accuracy for',nn,'nearest neighbors:',accuracy_score(labels[-n_test:],preds_wmd))

print('========== performance of centroids ==========')

for nn in [1,3,5,7,11,13,15,17,21,23]:
    
    preds_centroids = []
    for idx,sims in enumerate(my_centroid_similarities):
        idxs_sorted = np.argsort(sims).tolist()
        ### fill gap ### get labels of 'nn' nearest neighbors. Be cautious about the difference between distance and similarity!
        labels_nn = [labels[:n_train][elt] for elt in idxs_sorted[:nn]]
        # select most frequent label as prediction
        counts = dict(Counter(labels_nn))
        max_counts = max(list(counts.values()))
        pred = [k for k,v in counts.items() if v==max_counts][0]
        preds_centroids.append(pred)
    
    # compare predictions to true labels
    
    print('accuracy for',nn,'nearest neighbors:',accuracy_score(labels[-n_test:],preds_centroids))


print('========== performance of TFIDF ==========')

for nn in [1,3,5,7,11,13,15,17,21,23]:
    
    preds_tfidf = []
    for idx,sims in enumerate(my_similarities):
        # sort by decreasing order
        idxs_sorted = np.argsort(sims).tolist()
        ### fill gap ### get labels of 'nn' nearest neighbors. Be cautious about the difference between distance and similarity!
        labels_nn = [labels[:n_train][elt] for elt in idxs_sorted[:nn]]
        # select most frequent label as prediction
        counts = dict(Counter(labels_nn))
        max_counts = max(list(counts.values()))
        pred = [k for k,v in counts.items() if v==max_counts][0]
        preds_tfidf.append(pred)
    
    # compare predictions to true labels
    
    print('accuracy for',nn,'nearest neighbors:',accuracy_score(labels[-n_test:],preds_tfidf))
