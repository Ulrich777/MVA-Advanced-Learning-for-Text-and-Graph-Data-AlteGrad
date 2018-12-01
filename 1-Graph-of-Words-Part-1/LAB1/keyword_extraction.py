import os
import random
import string
import re 
import itertools
import operator
import copy
import networkx as nx
import nltk
from nltk.corpus import stopwords
# requires nltk >= 3.2.1
from nltk import pos_tag
# might also be required:
#nltk.download('maxent_treebank_pos_tagger')
#nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer

# import custom functions
from library import clean_text_simple, terms_to_graph, accuracy_metrics, weighted_core_dec

stemmer = nltk.stem.PorterStemmer()
stpwds = stopwords.words('english')
punct = string.punctuation.replace('-', '')

##################################
# read and pre-process abstracts #
##################################

path_to_abstracts = "C:/Users/utilisateur/Desktop/AlteGrad/ALTEGRAD/for_moodle 2/data/Hulth2003testing/abstracts"
abstract_names = sorted(os.listdir(path_to_abstracts))

abstracts = []
for counter,filename in enumerate(abstract_names):
    # read file
    with open(path_to_abstracts + '/' + filename, 'r') as my_file: 
        text = my_file.read().splitlines()
    text = ' '.join(text)
    # remove formatting
    text = re.sub('\s+', ' ', text)
    abstracts.append(text)
    # print progress
    if counter % round(len(abstract_names)/10) == 0:
        print(counter, 'files processed')

abstracts_cleaned = []
for counter,abstract in enumerate(abstracts):
    my_tokens = clean_text_simple(abstract,my_stopwords=stpwds,punct=punct)
    abstracts_cleaned.append(my_tokens)
    # print progress
    if counter % round(len(abstracts)/10) == 0:
        print(counter, 'abstracts processed')
                       
###############################################
# read and pre-process gold standard keywords #
###############################################

path_to_keywords = "C:/Users/utilisateur/Desktop/AlteGrad/ALTEGRAD/for_moodle 2/data/Hulth2003testing/uncontr"
keyword_names = sorted(os.listdir(path_to_keywords))
   
keywords_gold_standard = []

for counter,filename in enumerate(keyword_names):
    # read file
    with open(path_to_keywords + '/' + filename, 'r') as my_file: 
        text = my_file.read().splitlines()
    text = ' '.join(text)
    text =  re.sub('\s+', ' ', text) # remove formatting
    text = text.lower()
    # turn string into list of keywords, preserving intra-word dashes 
    # but breaking n-grams into unigrams
    keywords = text.split(';')
    keywords = [keyword.strip().split(' ') for keyword in keywords]
    keywords = [keyword for sublist in keywords for keyword in sublist] # flatten list
    keywords = [keyword for keyword in keywords if keyword not in stpwds] # remove stopwords (rare but can happen due to n-gram breaking)
    keywords_stemmed = [stemmer.stem(keyword) for keyword in keywords]
    keywords_stemmed_unique = list(set(keywords_stemmed)) # remove duplicates (can happen due to n-gram breaking)
    keywords_gold_standard.append(keywords_stemmed_unique)
    
    # print progress
    if counter % round(len(keyword_names)/10) == 0:
        print(counter, 'files processed')

#abstracts_cleaned[0]
#keywords_gold_standard[0]

###############################
# keyword extraction with gow #
###############################

keywords_gow = []  

for counter,abstract in enumerate(abstracts_cleaned):
    # create graph-of-words
    g = terms_to_graph(abstract, w=4)
    # decompose graph-of-words
    core_numbers =  unweighted_k_core(g)   #dict(zip(g.vs['name'],g.coreness()))
    # retain main core as keywords
    max_c_n = max(core_numbers.values())
    keywords = [kwd for kwd,c_n in core_numbers.items() if c_n==max_c_n]
    # save results
    keywords_gow.append(keywords)
    
    # print progress
    if counter % round(len(abstracts_cleaned)/10) == 0:
        print(counter, 'abstracts processed')

keywords_gow_w = []  

for counter,abstract in enumerate(abstracts_cleaned):
    # create graph-of-words
    g = terms_to_graph(abstract, w=4)
    # decompose graph-of-words
    core_numbers =  weighted_core_dec(g)   #dict(zip(g.vs['name'],g.coreness()))
    # retain main core as keywords
    max_c_n = max(core_numbers.values())
    keywords = [kwd for kwd,c_n in core_numbers.items() if c_n==max_c_n]
    # save results
    keywords_gow_w.append(keywords)
    
    # print progress
    if counter % round(len(abstracts_cleaned)/10) == 0:
        print(counter, 'abstracts processed')


#########################################
# keyword extraction with the baselines #
#########################################

# TF_IDF

my_percentage = 0.33

# to ensure same pre-processing as the other methods
abstracts_cleaned_strings = [' '.join(elt) for elt in abstracts_cleaned]

tfidf_vectorizer = TfidfVectorizer(stop_words=stpwds) # is stpwds necessary here?
doc_term_matrix = tfidf_vectorizer.fit_transform(abstracts_cleaned_strings)
### fill the gap (create an object 'terms' containing the column names of 'doc_term_matrix') ###
### hint: use the .get_feature_names() method ###
terms = tfidf_vectorizer.get_feature_names()

vectors_list = doc_term_matrix.todense().tolist()

keywords_tfidf = []

for counter,vector in enumerate(vectors_list):
    # bow feature vector as list of tuples
    terms_weights = zip(terms,vector)
    # keep only non zero values (the words in the document)
    ### fill the gap (create object 'nonzero') ###
    nonzero = [(term, score) for term, score in terms_weights if score>0]
    # rank by decreasing weights
    nonzero = sorted(nonzero, key=operator.itemgetter(1), reverse=True)
    # retain top 'my_percentage' words as keywords
    numb_to_retain = int(len(nonzero)*my_percentage)
    keywords = [tuple[0] for tuple in nonzero[:numb_to_retain]]
    
    keywords_tfidf.append(keywords)
    
    # print progress
    if counter % round(len(vectors_list)/10) == 0:
        print(counter, 'vectors processed')

# PageRank

keywords_pr = []

for counter,abstract in enumerate(abstracts_cleaned):
    ### fill the gaps ###
    ### hint: combine the beginning of the gow loop with the middle section of the tfidf loop ###
    ### use the .pagerank() igraph method ###
    g = terms_to_graph(abstract, w=4)
    pr = nx.pagerank(g)
    nonzero = [(term, score) for term, score in pr.items() if score>0]
    nonzero = sorted(nonzero, key=operator.itemgetter(1), reverse=True)
    numb_to_retain = int(len(nonzero)*my_percentage)
    keywords = [tuple[0] for tuple in nonzero[:numb_to_retain]]
    keywords_pr.append(keywords)
    if counter % round(len(abstracts_cleaned)/10) == 0:
        print(counter, 'abstracts processed')

##########################
# performance evaluation #
##########################

perf_gow = []
perf_gow_w = []
perf_tfidf = []
perf_pr = []

for idx, truth in enumerate(keywords_gold_standard):
    perf_gow.append(accuracy_metrics(keywords_gow[idx], truth))
    perf_gow_w.append(accuracy_metrics(keywords_gow_w[idx], truth))
    perf_tfidf.append(accuracy_metrics(keywords_tfidf[idx], truth))
    perf_pr.append(accuracy_metrics(keywords_pr[idx], truth))

lkgs = len(keywords_gold_standard)

# macro-averaged results (averaged at the collection level)

results = {'gow':perf_gow,'gow_w':perf_gow_w,'tfidf':perf_tfidf,'pr':perf_pr}

for name, result in results.items():
    print(name + ' performance: \n')
    print('precision:', sum([tuple[0] for tuple in result])/lkgs)
    print('recall:', sum([tuple[1] for tuple in result])/lkgs)
    print('F-1 score:', sum([tuple[2] for tuple in result])/lkgs)
    print('\n')
