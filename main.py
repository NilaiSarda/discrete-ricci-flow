import util
import networkx as nx
import numpy as np

corpus = [' '.join(sent) for sent in brown.sents()]
cleaned = util.clean(corpus)
print("Cleaned")
preprocessed = util.preprocess(cleaned)
print("Preprocessed")
docs, count, nouns = preprocessed
print(len(count), len(nouns))

words = {word : n for word, n in count.items()}
print(len(words))
inds = {word : i for i, word in enumerate(words.keys())}
inv = {i : word for word, i in inds.items()}
N = len(inds)
A = np.zeros((N, N))
B = np.zeros((N, N))
for doc in docs:
    doc = [word for word in doc if word in words]
    docind = [inds[word] for word in doc]
    for i in docind:
        for j in docind:
            A[i][j] += 1
        B[i][:] += 1

print("Processed text")       
M = np.zeros((N, N))
M[A>0] = A[A>0] * np.log(np.sum(B[0]) * A[A>0] / (B[A>0] + B.T[A>0]))
print(np.max(M), np.min(M), np.mean(M[M > 0]), np.median(M[M > 0]))
G = nx.from_numpy_matrix(M)
print(len(G.nodes), len(G.edges))
nx.write_gexf(G, 'brown_all.gexf')

sorted_n = sorted(((value, key) for (key,value) in nouns.items()), reverse=True)
print(sorted_n[:100])

import random
import pickle

cws, scs, mcs, rfs = [], [], [], []
for tt in tqdm(range(1)):
    #iu, iv = random.randint(3000, 6000), random.randint(8000, 12000)
    iu, iv = 10000, 10001
    wu, wv = sorted_n[iu][1], sorted_n[iv][1]
    u, v = inds[wu], inds[wv]
    Gu, Gv = nx.ego_graph(G, u), nx.ego_graph(G, v)
    Cu, Cv = set(Gu.nodes.keys()), set(Gv.nodes.keys())
    Cuv = Cu & Cv
    wuv = wu + "_" + wv
    M[u, :] += M[v, :]
    M[:, u] += M[:, v]
    np.delete(M, (v), axis=0)
    np.delete(M, (v), axis=1)
    H = nx.from_numpy_matrix(M)
    uv = u
    inv[uv] = wuv
    print(wuv)
    Wu, Wv = set(inv[k] for k in Cu), set(inv[k] for k in Cv)
    Wuv = Wu & Wv

    kill = set([uv])
    Guv = nx.ego_graph(H, uv)
    cls = [0, 0, 0, 0]

    for k, v in (Guv.edges.items()):
        Guv.edges[k]['weight'] *= (Guv.degree(k[0]) + Guv.degree(k[1]))
        Guv.edges[k]['weight'] = np.log(Guv.edges[k]['weight'])
        
    for node in (Guv.nodes):
        if inv[node] in Wuv:
            Guv.nodes[node]['color'] = 'purple'
            cls[0] += 1
        elif inv[node] in Wu:
            Guv.nodes[node]['color'] = 'red'
            cls[1] += 1
        elif inv[node] in Wv:
            Guv.nodes[node]['color'] = 'blue'
            cls[2] += 1
        else:
            kill.add(node)
            cls[3] += 1
    for node in kill:
        Guv.remove_node(node)
    
    visualize(Guv, "testfigs/"+wuv+".png", color=True)
    visualize(Gu, "testfigs/"+wu+".png")
    visualize(Gv, "testfigs/"+wv+".png")

    cw_s, sc_s, mc_s = score(Guv)
    cws.append(cw_s)
    scs.append(sc_s)
    mcs.append(mc_s)

    Guv_p = flow(Guv, wuv)
    y_ = naive_pred(Guv_p)
    y = get_ground_truth(Guv_p)
    print(word_clusters(Guv_p))
    rf_s = (*scores(y, y_), nmi(y, y_))
    print(rf_s)
    rfs.append(rf_s)
    if tt % 5 == 0:
        with open("tmp_{0}.pickle".format(tt), "wb") as f:
            pickle.dump((rfs,cws,scs,mcs), f)

print(rfs)
print(cws)
print(scs)
print(mcs)