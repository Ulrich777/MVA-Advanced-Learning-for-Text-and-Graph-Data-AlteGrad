import re 
import itertools
import operator
import copy
import networkx as nx
import heapq
import nltk
from nltk.corpus import stopwords
# requires nltk 3.2.1
from nltk import pos_tag

def clean_text_simple(text, my_stopwords, punct, remove_stopwords=True, pos_filtering=True, stemming=True):
    text = text.lower()
    text = ''.join(l for l in text if l not in punct) # remove punctuation (preserving intra-word dashes)
    text = re.sub(' +',' ',text) # strip extra white space
    text = text.strip() # strip leading and trailing white space
    # tokenize (split based on whitespace)
    ### fill the gap (store results as 'tokens') ###
    tokens = text.split(' ')
    if pos_filtering == True:
        # POS tag and retain only nouns and adjectives
        tagged_tokens = pos_tag(tokens)
        tokens_keep = []
        for item in tagged_tokens:
            if (
            item[1] == 'NN' or
            item[1] == 'NNS' or
            item[1] == 'NNP' or
            item[1] == 'NNPS' or
            item[1] == 'JJ' or
            item[1] == 'JJS' or
            item[1] == 'JJR'
            ):
                tokens_keep.append(item[0])
        tokens = tokens_keep
    if remove_stopwords:
        # remove stopwords from 'tokens'
        ### fill the gap ###
        tokens = [token for token in tokens if token not in my_stopwords]
    if stemming:
        # apply Porter's stemmer
        stemmer = nltk.stem.PorterStemmer()
        tokens_stemmed = list()
        for token in tokens:
            tokens_stemmed.append(stemmer.stem(token))
        tokens = tokens_stemmed
    
    return(tokens)


 
def terms_to_graph(terms, w):
    '''This function returns a directed, weighted igraph from a list of terms (the tokens from the pre-processed text) e.g., ['quick','brown','fox'].
    Edges are weighted based on term co-occurence within a sliding window of fixed size 'w'.
    '''
    
    from_to = {}
    
    # create initial complete graph (first w terms)
    terms_temp = terms[0:w]
    indexes = list(itertools.combinations(range(w), r=2))
    
    new_edges = []
    
    for my_tuple in indexes:
        new_edges.append(tuple([terms_temp[i] for i in my_tuple]))
    
    for new_edge in new_edges:
        if new_edge[0]!=new_edge[1]:
            if new_edge in from_to:
                from_to[new_edge] += 1
            else:
                from_to[new_edge] = 1

    # then iterate over the remaining terms
    for i in range(w, len(terms)):
        considered_term = terms[i] # term to consider
        terms_temp = terms[(i-w+1):(i+1)] # all terms within sliding window
        
        # edges to try
        candidate_edges = []
        for p in range(w-1):
            candidate_edges.append((terms_temp[p],considered_term))
    
        for try_edge in candidate_edges:
            
            if try_edge[1] != try_edge[0]:
            # if not self-edge
            
                # if edge has already been seen, update its weight
                ### fill the gap ###
                if try_edge in from_to:
                    from_to[try_edge] +=1
                                   
                # if edge has never been seen, create it and assign it a unit weight     
                else:
                    ### fill the gap ###
                    from_to[try_edge] = 1
    
    # create empty graph
    g = nx.DiGraph()
    
    # add vertices
    raw_data = [(k[0], k[1], v) for k,v in from_to.items()]
    
    # add edges, direction is preserved since the graph is directed
    g.add_weighted_edges_from(raw_data)
    
    
    return(g)



def unweighted_k_core(g):
    # work on clone of g to preserve g 
    gg = g.copy()    
    
    # initialize dictionary that will contain the core numbers
    cores_g = dict(zip(list(g.nodes),[0]*len(list(g.nodes))))
    
    i = 0
    
    # while there are vertices remaining in the graph
    while len(list(gg.nodes))>0:
        ### fill the gaps ###
        while [d for v,d in gg.out_degree() if d<=i]:
            #retieve the 
            node = [v for v,d in gg.out_degree() if d<=i][0]
            cores_g[node] = i
            gg.remove_node(node)
        
        i += 1
    
    return cores_g


   
def weighted_core_dec(g):
    '''
    k-core decomposition for weighted graphs (generalized k-cores)
    based on Batagelj and Zaversnik's (2002) algorithm #4
    '''
    # work on clone of g to preserve g 
    gg = g.copy()   
    # initialize dictionary that will contain the core numbers
    cores_g = dict(zip(list(gg.nodes),[0]*len(list(gg.nodes))))

    # initialize min heap of degrees
    PV = {node:weight for node, weight in gg.out_degree(weight='weight')}
    heap_g = list(zip(list(PV.values()), list(PV.keys())))
    heapq.heapify(heap_g)

    while len(heap_g) > 0:
        
        top = heap_g[0][1]
        # save names of its neighbors
        neighbors_top = list(gg.neighbors(top))
        # exclude self-edges
        neighbors_top = [elt for elt in neighbors_top if elt!=top]
        # set core number of heap top element as its weighted degree
        cores_g[top] = PV[top]
        # delete top vertex
        gg.remove_node(top)
        
        PV.pop(top)
        
        if len(neighbors_top)>0:
        # iterate over neighbors of top element
            for i, name_n in enumerate(neighbors_top):
                ### fill the gap (store result as 'max_n') ###
                max_n = max(cores_g[top], gg.out_degree(name_n, weight='weight'))
                PV[name_n] = max_n
                # update heap
                heap_g = list(zip(list(PV.values()), list(PV.keys())))
                heapq.heapify(heap_g)
        else:
            # update heap
            #PV.pop(top)
            heap_g = list(zip(list(PV.values()), list(PV.keys())))
            heapq.heapify(heap_g)
            
    # sort vertices by decreasing core number
    sorted_cores_g = dict(sorted(cores_g.items(), key=operator.itemgetter(1), reverse=True))
    
    return(sorted_cores_g)


def accuracy_metrics(candidate, truth):
    
    # true positives ('hits') are both in candidate and in truth
    tp = len(set(candidate).intersection(truth))
    
    # false positives ('false alarms') are in candidate but not in truth
    fp = len([element for element in candidate if element not in truth])
    
    # false negatives ('misses') are in truth but not in candidate
    fn = len([element for element in truth if element not in candidate])
    
    # precision
    prec = round(float(tp)/(tp+fp),5)
    
    # recall
    rec = round(float(tp)/(tp+fn),5)
    
    if prec+rec != 0:
        # F1 score
        f1 = round(2 * float(prec*rec)/(prec+rec),5)
    else:
        f1 = 0
       
    return (prec, rec, f1)
    

