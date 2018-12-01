import os
os.chdir("C:/Users/utilisateur/Desktop/LAST_YEAR/AlteGrad/ALTEGRAD/LAB2/code")


import math
import numpy
import pandas as pd
from library import terms_to_graph, compute_node_centrality, print_top10, print_bot10
from sklearn import svm, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

##################################
# data loading and preprocessing #
##################################

# add your absolute path here
path_to_data = "C:/Users/utilisateur/Desktop/LAST_YEAR/AlteGrad/ALTEGRAD/LAB2/data"



train = pd.read_csv(path_to_data + "/webkb-train-stemmed.txt", header=None, delimiter="\t")
print(train.shape)

test = pd.read_csv(path_to_data + "/webkb-test-stemmed.txt", header=None, delimiter="\t")
print(test.shape)

# inspect head of data frames
print("first five rows of training data:")
print(train.iloc[:5,:])

print("first five rows of testing data:")
print(test.iloc[:5,:])

# get index of empty (nan) and less than four words documents (for which a graph cannot be built)
index_remove = [i for i in range(len(train.iloc[:,1])) if (train.iloc[i,1]!=train.iloc[i,1]) or ((train.iloc[i,1]==train.iloc[i,1])and(len(train.iloc[i,1].split(" "))<4))]

# remove those documents
print("removing", len(index_remove), "documents from training set")
train = train.drop(train.index[index_remove])
print(train.shape)

# repeat above steps for test set
index_remove = [i for i in range(len(test.iloc[:,1])) if (test.iloc[i,1]!=test.iloc[i,1]) or ((test.iloc[i,1]==test.iloc[i,1])and(len(test.iloc[i,1].split(" "))<4))]
print("removing", len(index_remove), "documents from test set")
test = test.drop(test.index[index_remove])
print(test.shape)

labels = train.iloc[:,0]
unique_labels = list(set(labels))

truth = test.iloc[:,0]
unique_truth = list(set(truth))

print("number of observations per class:")
for label in unique_labels:
    print(label, ":", len([temp for temp in labels if temp==label]))

print("storing terms from training documents as list of lists")
terms_by_doc = [document.split(" ") for document in train.iloc[:,1]]
n_terms_per_doc = [len(terms) for terms in terms_by_doc]

print("storing terms from test documents as list of lists")
terms_by_doc_test = [document.split(" ") for document in test.iloc[:,1]]

print("min, max and average number of terms per document:", min(n_terms_per_doc), max(n_terms_per_doc), sum(n_terms_per_doc)/len(n_terms_per_doc))

# store all terms in list
### fill the gap ### hint: flatten 'terms_by_doc' (you may use a list comprehension)
all_terms = [term for doc in terms_by_doc for term in doc]

# compute average number of terms
avg_len = sum(n_terms_per_doc)/len(n_terms_per_doc)

# unique terms
### fill the gap ### hint: get unique elements of 'all_terms'
all_unique_terms = list(set(all_terms))

# store IDF values in dictionary
terms_by_doc_sets = [set(elt) for elt in terms_by_doc]
n_doc = len(labels)
idf = dict(zip(all_unique_terms,[0]*len(all_unique_terms)))

for counter,unique_term in enumerate(list(idf.keys())):
    # compute number of documents in which 'unique_term' appears
    ### fill the gap ### 
    df = sum([unique_term in terms_by_doc_set for terms_by_doc_set in terms_by_doc_sets])
    #hint: iterate over 'terms_by_doc_sets' and test for the presence of 'unique_term' (you may use a list comprehension). You'll get a list of booleans. Sum it to get the counts
    # idf
    ### fill the gap ### hint: use math.log10 and refer to the beginning of Section 2 in the handout
    idf[unique_term] = math.log10((train.shape[0]+1)/df)
    if counter % 1e3 == 0:
        print(counter, "terms processed")

###########################################
# computing features for the training set #
###########################################

w = 3 # sliding window size

print("creating a graph-of-words for the collection")

### fill the gap ### hint: use the terms_to_graph function with the proper arguments
c_g = terms_to_graph(terms_by_doc ,w, overspanning=False)

# sanity check (should return True)
print(len(all_unique_terms) == c_g.number_of_nodes())

print("creating a graph-of-words for each training document")

all_graphs = []
for elt in terms_by_doc:
    all_graphs.append(terms_to_graph([elt],w,overspanning=True))

# sanity checks (should return True)
print(len(terms_by_doc)==len(all_graphs))
print(len(set(terms_by_doc[0]))==all_graphs[0].number_of_nodes())

print("computing vector representations of each training document")

b = 0.003

features_degree = []
features_w_degree = []
features_closeness = []
features_w_closeness = []
features_twicw = [] # we try it only with unweighted degree
features_tfidf = []

len_all = len(all_unique_terms)
### fill the gap ### hint: build a dict where the keys are the names of the nodes in the collection graph and the values are their unweighted degrees
collection_degrees = dict(c_g.out_degree())
maxcol = max(list(collection_degrees.values()))

for i, graph in enumerate(all_graphs):
    
    terms_in_doc = terms_by_doc[i]
    doc_len = len(terms_in_doc)
    
    # returns node (0) name, (1) degree, (2) weighted degree, (3) closeness, (4) weighted closeness
    my_metrics = compute_node_centrality(graph)
    
    feature_row_degree = [0]*len_all
    feature_row_w_degree = [0]*len_all
    feature_row_closeness = [0]*len_all
    feature_row_w_closeness = [0]*len_all
    feature_row_twicw = [0]*len_all
    feature_row_tfidf = [0]*len_all
    
    # iterate over the unique terms contained by the doc (for all the other columns, the values will remain at zero)
    for term in list(set(terms_in_doc)):
        
        index = all_unique_terms.index(term)
        idf_term = idf[term]
        denominator = 1-b+(b*(float(doc_len)/avg_len))
        ### fill the gap ### hint: refer to the TF equation in the handout
        metrics_term = [tuple_[1:] for tuple_ in my_metrics if tuple_[0]==term][0]
        
        # store TW-IDF values
        feature_row_degree[index] = (metrics_term[0]/denominator) * idf_term
        feature_row_w_degree[index] = (metrics_term[1]/denominator) * idf_term
        feature_row_closeness[index] = (metrics_term[2]/denominator) * idf_term
        feature_row_w_closeness[index] = (metrics_term[3]/denominator) * idf_term
        
        # store TW-ICW values
        feature_row_twicw[index] = (metrics_term[0]/denominator) * math.log10((maxcol+1)/(collection_degrees[term]+0.01)) 
        
        # number of occurences of word in document
        tf = terms_in_doc.count(term)        
        # store TF-IDF value
        feature_row_tfidf[index] = ((1+math.log1p(1+math.log1p(tf)))/(1-0.2+(0.2*(float(doc_len)/avg_len)))) * idf_term
    
    features_degree.append(feature_row_degree)
    features_w_degree.append(feature_row_w_degree)
    features_closeness.append(feature_row_closeness)
    features_w_closeness.append(feature_row_w_closeness)
    features_twicw.append(feature_row_twicw)
    features_tfidf.append(feature_row_tfidf)

    if i % 1000 == 0:
        print (i, "documents processed")

# convert list of lists into array
# documents as rows, unique words (features) as columns
training_set_degree = numpy.array(features_degree)
training_set_w_degree = numpy.array(features_w_degree)
training_set_closeness = numpy.array(features_closeness)
training_set_w_closeness = numpy.array(features_w_closeness)
training_set_tw_icw = numpy.array(features_twicw)
training_set_tfidf = numpy.array(features_tfidf)

#######################################
# computing features for the test set #
#######################################

print("creating a graph-of-words for each test document")

all_graphs_test = []
for elt in terms_by_doc_test:
    all_graphs_test.append(terms_to_graph([elt],w,overspanning=True))

# sanity checks (should return True)
print(len(terms_by_doc_test)==len(all_graphs_test))
print(len(set(terms_by_doc_test[0]))==all_graphs_test[0].number_of_nodes())

print("computing vector representations of each test document")
# ! each test document is represented in the training space only

features_degree_test = []
features_w_degree_test = []
features_closeness_test = []
features_w_closeness_test = []
features_twicw_test = []
features_tfidf_test = []

for i, graph in enumerate(all_graphs_test):
    
    # filter out the terms that are not in the training set
    terms_in_doc = [term for term in terms_by_doc_test[i] if term in all_unique_terms]
    doc_len = len(terms_in_doc)
    
    my_metrics = compute_node_centrality(graph)
    
    feature_row_degree_test = [0]*len_all
    feature_row_w_degree_test = [0]*len_all
    feature_row_closeness_test = [0]*len_all
    feature_row_w_closeness_test = [0]*len_all
    feature_row_twicw_test = [0]*len_all
    feature_row_tfidf_test = [0]*len_all

    for term in list(set(terms_in_doc)):
        index = all_unique_terms.index(term)
        idf_term = idf[term]
        denominator = (1-b+(b*(float(doc_len)/avg_len)))
        metrics_term = [tuple[1:] for tuple in my_metrics if tuple[0]==term][0]
        
        # store TW-IDF values      
        feature_row_degree_test[index] = (metrics_term[0]/denominator) * idf_term
        feature_row_w_degree_test[index] = (metrics_term[1]/denominator) * idf_term
        feature_row_closeness_test[index] = (metrics_term[2]/denominator) * idf_term
        feature_row_w_closeness_test[index] = (metrics_term[3]/denominator) * idf_term
        
        # store TW-ICW values
        feature_row_twicw_test[index] = (metrics_term[0]/denominator) * (math.log10((maxcol+1)/(collection_degrees[term]+.01)))

        # number of occurences of word in document
        tf = terms_in_doc.count(term)
        # store TF-IDF value
        feature_row_tfidf_test[index] = ((1+math.log1p(1+math.log1p(tf)))/(1-0.2+(0.2*(float(doc_len)/avg_len)))) * idf_term

    features_degree_test.append(feature_row_degree_test)
    features_w_degree_test.append(feature_row_w_degree_test)
    features_closeness_test.append(feature_row_closeness_test)
    features_w_closeness_test.append(feature_row_w_closeness_test)
    features_twicw_test.append(feature_row_twicw_test)
    features_tfidf_test.append(feature_row_tfidf_test)
    
    if i % 500 == 0:
        print (i, "documents processed")

# convert list of lists into array
# documents as rows, unique words as columns (i.e., document-term matrix)
testing_set_degree = numpy.array(features_degree_test)
testing_set_w_degree = numpy.array(features_w_degree_test)
testing_set_closeness = numpy.array(features_closeness_test)
testing_set_w_closeness = numpy.array(features_w_closeness_test)
testing_set_twicw = numpy.array(features_twicw_test)
testing_set_tfidf = numpy.array(features_tfidf_test)

##########
# labels #
##########

# convert labels into integers then into column array
labels = list(labels)
labels_int = [0] * len(labels)
for j in range(len(unique_labels)):
    index_temp = [i for i in range(len(labels)) if labels[i]==unique_labels[j]]
    for element in index_temp:
        labels_int[element] = j
        
# convert truth into integers then into column array
truth = list(truth)
truth_int = [0] * len(truth)
for j in range(len(unique_truth)):
    index_temp = [i for i in range(len(truth)) if truth[i]==unique_truth[j]]
    for element in index_temp:
        truth_int[element] = j

# check that coding went smoothly
print(list(zip(truth_int,truth))[:20])

truth_array = numpy.array(truth_int)

# check that coding went smoothly
print(list(zip(labels_int,labels))[:20])
labels_array = numpy.array(labels_int)

for clf in ["LinearSVC","LogisticRegression","MultinomialNB"]:
    
    if clf=="LinearSVC":
        classifier_degree = svm.LinearSVC()
        classifier_w_degree = svm.LinearSVC()
        classifier_closeness = svm.LinearSVC()
        classifier_w_closeness = svm.LinearSVC()
        classifier_twicw = svm.LinearSVC()
        classifier_tfidf = svm.LinearSVC()
    elif clf=="LogisticRegression":
        classifier_degree = LogisticRegression(multi_class='ovr',solver='liblinear') # we specify multi_class and solver arguments just to avoid getting a warning
        classifier_w_degree = LogisticRegression(multi_class='ovr',solver='liblinear')
        classifier_closeness = LogisticRegression(multi_class='ovr',solver='liblinear')
        classifier_w_closeness = LogisticRegression(multi_class='ovr',solver='liblinear')
        classifier_twicw = LogisticRegression(multi_class='ovr',solver='liblinear')
        classifier_tfidf = LogisticRegression(multi_class='ovr',solver='liblinear')
    elif clf=="MultinomialNB":
        classifier_degree = MultinomialNB()
        classifier_w_degree = MultinomialNB()
        classifier_closeness = MultinomialNB()
        classifier_w_closeness = MultinomialNB()
        classifier_twicw = MultinomialNB()
        classifier_tfidf = MultinomialNB()
    
    ############
    # training #
    ############
    
    print("training", clf, "classifiers")
    classifier_degree.fit(training_set_degree, labels_array)
    classifier_w_degree.fit(training_set_w_degree, labels_array)
    classifier_closeness.fit(training_set_closeness, labels_array)
    classifier_w_closeness.fit(training_set_w_closeness, labels_array)
    classifier_twicw.fit(training_set_tw_icw, labels_array)
    classifier_tfidf.fit(training_set_tfidf, labels_array)
    
    ###########
    # testing #
    ###########
    
    # issue predictions
    predictions_degree = classifier_degree.predict(testing_set_degree)
    predictions_w_degree = classifier_w_degree.predict(testing_set_w_degree)
    predictions_closeness = classifier_closeness.predict(testing_set_closeness)
    predictions_w_closeness = classifier_w_closeness.predict(testing_set_w_closeness)
    predictions_twicw = classifier_twicw.predict(testing_set_twicw)
    predictions_tfidf = classifier_tfidf.predict(testing_set_tfidf)
    
    print('========== accuracy for', clf ,'classifier ==========')
    print("accuracy TW-IDF degree:", round(metrics.accuracy_score(truth_array,predictions_degree)*100,3))
    print("accuracy TW-IDF weighted degree:", round(metrics.accuracy_score(truth_array,predictions_w_degree)*100,3))
    print("accuracy TW-IDF closeness:", round(metrics.accuracy_score(truth_array,predictions_closeness)*100,3))
    print("accuracy TW-IDF weighted closeness:", round(metrics.accuracy_score(truth_array,predictions_w_closeness)*100,3))
    print("accuracy TW-ICW degree:", round(metrics.accuracy_score(truth_array,predictions_twicw)*100,3))
    print("accuracy TF-IDF:", round(metrics.accuracy_score(truth_array,predictions_tfidf)*100,3))
    
# show the most and less important features for each class
### fill the gaps ### hint: pick a classifier (e.g., 'classifier_tfidf'), and pass it to the 'print_top10' and 'print_bot10' functions along with 'unique_labels' and 'all_unique_terms'

print_bot10(all_unique_terms, classifier_degree, unique_labels)
print_top10(all_unique_terms, classifier_degree, unique_labels)

