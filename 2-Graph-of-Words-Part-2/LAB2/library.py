import itertools
import networkx as nx
import numpy as np

def terms_to_graph(lists_of_terms, window_size, overspanning):
    '''This function returns a directed, weighted igraph from lists of list of terms (the tokens from the pre-processed text)
    e.g., [['quick','brown','fox'], ['develop', 'remot', 'control'], etc]
    Edges are weighted based on term co-occurence within a sliding window of fixed size 'w' '''

    if overspanning:
        terms = [item for sublist in lists_of_terms for item in sublist]
    else:
        idx = 0
        terms = lists_of_terms[idx]

    from_to = {}

    while True:
        w = min(window_size, len(terms))
        # create initial complete graph (first w terms)
        terms_temp = terms[0:w]
        indexes = list(itertools.combinations(range(w), r=2))

        new_edges = []

        for my_tuple in indexes:
            new_edges.append(tuple([terms_temp[i] for i in my_tuple]))
        for new_edge in new_edges:
            if new_edge in from_to:
                from_to[new_edge] += 1
            else:
                from_to[new_edge] = 1

        # then iterate over the remaining terms
        for i in range(w, len(terms)):
            # term to consider
            considered_term = terms[i]
            # all terms within sliding window
            terms_temp = terms[(i - w + 1):(i + 1)]

            # edges to try
            candidate_edges = []
            for p in range(w - 1):
                candidate_edges.append((terms_temp[p], considered_term))

            for try_edge in candidate_edges:

                # if not self-edge
                if try_edge[1] != try_edge[0]:

                    # if edge has already been seen, update its weight
                    if try_edge in from_to:
                        from_to[try_edge] += 1

                    # if edge has never been seen, create it and assign it a unit weight
                    else:
                        from_to[try_edge] = 1

        if overspanning:
            break
        else:
            idx += 1
            if idx == len(lists_of_terms):
                break
            terms = lists_of_terms[idx]

    # create empty graph
    g = nx.DiGraph()

    # add vertices
    if overspanning:
        g.add_nodes_from(terms)
        raw_data = [(k[0], k[1], v) for k,v in from_to.items()]
        g.add_weighted_edges_from(raw_data)
    else:
        raw_data = [(k[0], k[1], v) for k,v in from_to.items()]
        g.add_weighted_edges_from(raw_data)


    return (g)
    
def compute_node_centrality(graph):
    # degree
    degrees = dict(graph.out_degree())

    # weighted degree
    ### fill the gap ### hint: use the .strength() method with 'weights' argument
    w_degrees = dict(graph.out_degree(weight='weight'))

    # closeness
    ### fill the gap ### hint: use the .closeness() method with 'normalized' argument set to True
    closeness = nx.closeness_centrality(graph)

    # weighted closeness
    ### fill the gap ### hint: same as above, but with 'weights' argument
    w_closeness = nx.closeness_centrality(graph, distance='weight')
    
    out = [(n, degrees[n]/(graph.number_of_nodes()-1), w_degrees[n]/(graph.number_of_nodes()-1), 
            closeness[n], w_closeness[n]) for n in graph.nodes()]

    return out


def print_top10(feature_names, clf, class_labels):
    """Prints features with the highest coefficient values, per class"""
    # coef stores the weights of each feature (in unique term), for each class
    for i, class_label in enumerate(class_labels):
        top10 = np.argsort(clf.coef_[i])[-10:]
        print("%s: %s" % (class_label," ".join(feature_names[j] for j in top10)))

def print_bot10(feature_names, clf, class_labels):
    """Prints features with the lowest coefficient values, per class"""
    for i, class_label in enumerate(class_labels):
        bot10 = np.argsort(clf.coef_[i])[0:9]
        print("%s: %s" % (class_label," ".join(feature_names[j] for j in bot10)))
