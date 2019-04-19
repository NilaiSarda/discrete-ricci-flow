import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import spacy
import re
import markov_clustering as mc

from tqdm import tqdm
from collections import Counter
from nltk.corpus import brown
from nltk.corpus import stopwords
from nltk import tokenize

def visualize(G, name, color=False):
    for k in G.edges:
        G.edges[k]['spring_weight'] = G.edges[k]['weight']
    spring_pos = nx.spring_layout(G, weight='spring_weight')
    edge_color = [d['weight'] for (u, v, d) in G.edges(data=True)]
    if color:
        node_color = [d['color'] for (_, d) in G.nodes(data=True)]
        nx.draw_networkx(G, pos=spring_pos, with_labels=False, node_color=node_color, node_size=100, edge_color=edge_color)
    else:
        nx.draw_networkx(G, pos=spring_pos, with_labels=False, node_size=100, edge_color=edge_color)
    plt.show()
    plt.savefig(name)
    plt.clf()

from sklearn.cluster import AffinityPropagation, SpectralClustering
from sklearn.metrics import *
from scipy.special import comb
from chinese_whispers import chinese_whispers

def scores(y, y_):
    y_ = y_[y > 0] - 1
    y = y[y > 0] - 1
    return homogeneity_completeness_v_measure(y, y_)

def nmi(y, y_):
    y_ = y_[y > 0] - 1
    y = y[y > 0] - 1
    return normalized_mutual_info_score(y, y_, average_method='arithmetic')

def score(G):
    FF = G.copy()
    for k, v in FF.edges.items():
        FF.edges[k]['weight'] = 1./FF.edges[k]['weight']
        
    M = nx.adjacency_matrix(FF).todense()
    N = M.shape[0]

    SC = SpectralClustering(n_clusters=2, affinity='precomputed')
    clustering = SC.fit(M)
    y_sc = clustering.labels_ + 1
    chinese_whispers(FF, iterations=100)
    
    result = mc.run_mcl(M, inflation=2)
    clusters = mc.get_clusters(result)
    y_mc = mc_pred(FF, clusters)
    
    y = np.zeros_like(clustering.labels_)
    y_cw = []
    for i, x in enumerate(FF.nodes):
        if FF.nodes[x]['color'] == 'blue':
            y[i] = 1
        elif FF.nodes[x]['color'] == 'red':
            y[i] = 2
        y_cw.append(FF.nodes[x]['label'])
    labels = {x: i for i, x in enumerate(set(y_cw))}
    y_cw = [labels[x] for x in y_cw]
    y_cw = np.array(y_cw)
    
    print(*scores(y, y_cw), nmi(y, y_cw))
    print(*scores(y, y_sc), nmi(y, y_sc))
    print(*scores(y, y_mc), nmi(y, y_mc))
    
    return ((*scores(y, y_cw), nmi(y, y_cw)), 
            (*scores(y, y_sc), nmi(y, y_sc)),
            (*scores(y, y_mc), nmi(y, y_mc)))

import math
import ot 

def flow(F, name):
    G = F.copy()
    for k, v in G.edges.items():
        G.edges[k]['weight'] = 1./G.edges[k]['weight']
    h = 0.1 # time step size
    N = 1000 # number of Ricci flow steps

    alpha = 0
    p = 0
    for t in range(N):
        geo_dists = dict(nx.shortest_path_length(G, weight='weight'))
        # compute the Ricci curvature of all edges
        for k,v in (G.edges.items()):
            # construct nearby distribution to start node of edge
            supp0 = [k[0]]
            rho0 = [alpha]
            for l,u in G.adj[k[0]].items():
                supp0.append(l)
                rho0.append(np.exp(-geo_dists[k[0]][l]**p))
            rho0 = rho0 / np.sum(rho0)
            # supp0.sort()

            # construct nearby distribution to end node of edge
            supp1 = [k[1]]
            rho1 = [alpha]
            for l,u in G.adj[k[1]].items():
                supp1.append(l)
                rho1.append(np.exp(-geo_dists[k[1]][l] ** p))
            rho1 = rho1 / np.sum(rho1)
            # supp1.sort()

            # extract cost submatrix
            n = rho0.size
            m = rho1.size
            cost = np.zeros((n,m))
            for i in range(n):
                for j in range(m):
                    cost[i,j] = geo_dists[supp0[i]][supp1[j]]
            # compute OT cost and edge Ricci curvature
            w1 = ot.emd2(rho0, rho1, cost)
            G.edges[k]['ricci'] = 1 - w1/geo_dists[k[0]][k[1]]
        #update the edge weights
        for k,v in G.edges.items():
            edgeRicci = G.edges[k]['ricci']
            G.edges[k]['weight'] = geo_dists[k[0]][k[1]] - h*edgeRicci * geo_dists[k[0]][k[1]]
        sorted_edges = sorted(G.edges.keys(), key=lambda x:G.edges[x]['weight'], reverse=True)
        n = len(sorted_edges)
        for e in sorted_edges[:int(n/30)]:
            u, v = e
            if len(G[u]) != 1 and len(G[v]) != 1:
                G.remove_edge(*e)
        visualize(G, name + ".{0}.png".format(t), color=True)
        ccs = [len(c) for c in nx.connected_components(G)]
    return G

def mc_pred(G, clusters):
    y = [1 for _ in G.nodes]
    solos = 0
    for i, node in enumerate(G.nodes):
        for j in range(len(clusters)):
            if i in clusters[j]:
                y[i] = j+1
                break
    return np.array(y)

def naive_pred(G):
    clusters = [list(c) for c in nx.connected_components(G)]
    y = []
    for node in G.nodes:
        for i in range(len(clusters)):
            if node in clusters[i]:
                y.append(i+1)
    return np.array(y)

def word_clusters(G):
    clusters = [list(c) for c in nx.connected_components(G)]
    for c in clusters:
        tmpwords = []
        for node in c:
            tmpwords.append(inv_m[node])
        print(tmpwords)
        
def get_ground_truth(G):
    y = [0 for _ in range(len(G.nodes))]
    for i, x in enumerate(G.nodes):
        if G.nodes[x]['color'] == 'blue':
            y[i] = 1
        elif G.nodes[x]['color'] == 'red':
            y[i] = 2
    return np.array(y)

