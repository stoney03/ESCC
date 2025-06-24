from collections import defaultdict
from tracemalloc import start
from tqdm import tqdm
from skimage.measure import regionprops
from scipy import stats
from scipy.spatial import cKDTree
from openslide import OpenSlide
import skimage.feature as skfeat
import cv2
import numpy as np
import igraph as ig
import json
import time
import os
import multiprocessing as mp
import xml.etree.ElementTree as et
import pandas as pd

try:
    mp.set_start_method('spawn')
except:
    pass


def getGraphDisKnnFeatures(name, disKnnList,distanceThreshold = 100):
    result = defaultdict(list)
    result['name'] = name
    # disKnnList[np.isinf(disKnnList)] = np.nan
    disKnnList[np.isinf(disKnnList)] = distanceThreshold
    disKnnList_valid = np.ma.masked_invalid(disKnnList)
    result['minEdgeLength'] += np.min(disKnnList_valid, axis=1).tolist()
    result['maxEdgeLength'] += np.max(disKnnList_valid, axis=1).tolist()
    result['meanEdgeLength'] += np.mean(disKnnList_valid, axis=1).tolist()
    result['stdEdgeLength'] += np.std(disKnnList_valid, axis=1).tolist()
    result['skewnessEdgeLength'] += stats.skew(np.array(disKnnList),axis=1, nan_policy='omit').tolist()
    result['kurtosisEdgeLength'] += stats.kurtosis(np.array(disKnnList), axis=1, nan_policy='omit').tolist()
    return result


def getSingleGraphFeatures(args):
    subgraph, cmd = args
    result = defaultdict(list)
    n = subgraph.vcount()
    if cmd == 'name':
        result['name'] += [int(i) for i in subgraph.vs['name']]
    elif cmd == 'Nsubgraph':
        result['Nsubgraph'] += [n] * n
    elif cmd == 'Degrees':
        result['Degrees'] += subgraph.degree()
    elif cmd == 'Eigenvector':
        result['Eigenvector'] += subgraph.eigenvector_centrality()  
    # Slow
    elif cmd == 'Closeness':
        result['Closeness'] += subgraph.closeness()
    # Slow
    elif cmd == 'Betweenness':
        betweenness = np.array(subgraph.betweenness())
        result['Betweenness'] += betweenness.tolist()
        if n != 1 and n != 2:
            betweenness = betweenness / ((n - 1) * (n - 2) / 2)
        result['Betweenness_normed'] += betweenness.tolist()
    elif cmd == 'AuthorityScore':
        result['AuthorityScore'] += subgraph.authority_score()
    elif cmd == 'Coreness':
        result['Coreness'] += subgraph.coreness()
    elif cmd == 'Diversity':
        result['Diversity'] += subgraph.diversity()
    # Slow
    elif cmd == 'Eccentricity' or cmd == 'Eccentricity_normed':
        eccentricity = np.array(subgraph.eccentricity())
        result['Eccentricity'] += eccentricity.tolist()
        result['Eccentricity_normed'] += (eccentricity / n).tolist()
    # Slow
    elif cmd == 'HarmonicCentrality':
        result['HarmonicCentrality'] += subgraph.harmonic_centrality()
    elif cmd == 'HubScore':
        result['HubScore'] += subgraph.hub_score()
    elif cmd == 'NeighborhoodSize':
        result['NeighborhoodSize'] += subgraph.neighborhood_size()
    elif cmd == 'Strength':
        result['Strength'] += subgraph.strength()
    elif cmd == 'ClusteringCoefficient':
        result['ClusteringCoefficient'] += subgraph.transitivity_local_undirected()
    elif cmd == 'DegreeCentrality':
        if n!=1:
            result['DegreeCentrality'] += [x / (n - 1) for x in subgraph.degree()]
        else:
            result['DegreeCentrality'] += subgraph.degree()
    elif cmd == 'ConstraintScore':
        result['ConstraintScore'] += subgraph.constraint()
    return result


def getGraphCenterFeatures(graph: ig.Graph):
    result = defaultdict(list)
    # norm_cmds = ['name', 'Nsubgraph', 'Eigenvector', 'Degrees', 'AuthorityScore', 'Coreness', 'Diversity',
    #             'HubScore', 'NeighborhoodSize', 'Strength', 'ClusteringCoefficient']
    norm_cmds = ['name', 'Nsubgraph', 'Degrees',
                 # 'AuthorityScore', 'HubScore', 'Eigenvector','Diversity','NeighborhoodSize','Strength','DegreeCentrality','ConstraintScore',#'Cocitation',#'PageRank',#'Assortativity',# 按需选取
                 'Coreness', 'ClusteringCoefficient']
    multi_cmds = ['Eccentricity', 'HarmonicCentrality', 'Closeness', 'Betweenness']
    for subgraph in tqdm(graph.decompose()):
        for cmd in norm_cmds:
            args = [subgraph, cmd]
            ans = getSingleGraphFeatures(args)
            for k, v in zip(ans.keys(), ans.values()):
                result[k] += v

        for cmd in multi_cmds:
            args = [subgraph, cmd]
            ans = getSingleGraphFeatures(args)
            for k, v in zip(ans.keys(), ans.values()):
                result[k] += v
    return result


def constructGraphFromDict(
        wsiPath: str, nucleusInfo: dict, distanceThreshold: float,
        knn_n: int = 5, level: int = 0, offset=np.array([0, 0])
):
    r"""Construct graph from nucleus information dictionary

    Parameters
    ----------
    nucleusInfo : dict
        'mag': int
            magnification of the result
        'nuc': dict
            nucleus information
            'nuclei ID' : dict
                note that ID generated from HoverNet is not continuous
                'bbox' : list
                    [[left, top], [right, bottom]]
                'centroid' : list
                    [column, row]
                'contour' : list, from cv2.findContours
                    [[column1, row1], [column2, row2], ... ]
                'type_prob' : float
                    The probability of current nuclei belonging to type 'type'
                'type' : int

    distanceThreshold : maximum distance in magnification of 40x

    "type_map": {"Background": 0, "Neoplastic": 1, "Inflammatory": 2, "Connective": 3, "Dead": 4, "Epithelial": 5}
    
    cellSize : int, odd
        size of cell cropped for extracting GLCM features

    level : int
        level for reading WSI
        0 : 40x
        1 : 20x
        ...

    Returns
    -------
    graph :

    """
    flag = True
    offset = np.array([0, 0])
    print(f"{'Total 9 steps: 0 ~ 8':*^30s}")
    mag = 40
    distanceThreshold = distanceThreshold / (40.0 / mag)

    bboxes, centroids, contours, types = [], [], [], []

    for nuc_id, nucInfo in tqdm(nucleusInfo['nuc'].items(),
                        desc="0. Preparing"):

        if nucInfo['contour'] and all(len(point) == 2 for point in nucInfo['contour']):

            tmpCnt = np.array(nucInfo['contour'])
            left, top = tmpCnt.min(0)
            right, bottom = tmpCnt.max(0)
            bbox = [[left + offset[0], top + offset[1]], [right + offset[0], bottom + offset[1]]]
            bboxes.append(bbox)  # [[[, ],[, ]], [[, ],[, ]], ......]
            centroids.append(nucInfo['centroid'])  ## [[, ], [, ], ......]
            contours.append(nucInfo['contour'])
            types.append(nucInfo['type'])  ## [, , , ......]
    assert len(bboxes) == len(centroids) == len(
        types), 'The attribute of nodes (bboxes, centroids, types) must have same length'
    vertex_len = len(bboxes)
    globalGraph = ig.Graph()
    names = [str(i) for i in range(vertex_len)]

    globalGraph.add_vertices(vertex_len, attributes={
        'name': names, 'Bbox': bboxes, 'Centroid': centroids,
        'Contour': contours, 'CellType': types})

    t3 = time.time()

    backgIDs, neoplaIDs, inflamIDs, connecIDs, deadIDs, epitheIDs = \
        [np.where(np.array(types) == i)[0].tolist() for i in range(6)] 
    edge_info = pd.DataFrame({'source': [], 'target': [], 'featype': []})
    # {'Background': 0, 'Neoplastic': 1, 'Inflammatory': 2, 'Connective': 3, 'Dead': 4, 'Epithelial': 5}
    
    # Neopla->T, Inflam->I, Connec->S, Normal->N
    featype_dict = {'T-T': [neoplaIDs, neoplaIDs],
                 
                    'I-I': [inflamIDs, inflamIDs],
                    'S-S': [connecIDs, connecIDs],
                    # 'N-N': [normalIDs, normalIDs],
                    'T-I': [neoplaIDs, inflamIDs],
                    'T-S': [neoplaIDs, connecIDs],
                    # 'T-N': [neoplaIDs, normalIDs],
                    'I-S': [inflamIDs, connecIDs],
                    # 'I-N': [inflamIDs, normalIDs],
                    # 'S-N': [connecIDs, normalIDs]
    }


    for featype, featype_index_list in zip(featype_dict.keys(),
                                           featype_dict.values()): 
        print(f'Getting {featype} graph feature')
        print(f'---Creating edges')
        # Treat neopla and normal as the same cell type by making the same cellTypeMark,
        # and delete the edge between vertexs which have the same cellTypeMark
        pairs = np.array([]).reshape((0, 2))
        disKnnList = np.array([]).reshape((0, knn_n))
        subgraph_names = []
        featype_index = []

        for index in featype_index_list:
            featype_index += index
        for src, i_src in enumerate(featype_index_list):
            for tar, i_tar in enumerate(featype_index_list):
                if src != tar:

                    print("patch_name", wsiPath)
                    print("globalGraph.induced_subgraph(i_tar).vs.attributes:",globalGraph.induced_subgraph(i_tar).vs.attributes())
                    print("globalGraph.induced_subgraph(i_src).vs.attributes:",globalGraph.induced_subgraph(i_src).vs.attributes())
                    centroid_tar = globalGraph.induced_subgraph(i_tar).vs['Centroid']
                    centroid_src = globalGraph.induced_subgraph(i_src).vs['Centroid']
                    
                    if centroid_tar == [] or centroid_src == []:
                        flag = False
                        return globalGraph, edge_info, flag
                    
                    n_tar = len(i_tar)
                    n_src = len(i_src)
                    tree = cKDTree(centroid_tar) # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query.html#scipy.spatial.cKDTree.query
                    if i_src == i_tar:
                        disknn, vertex_index = tree.query(centroid_src, k=knn_n + 1,
                                                          distance_upper_bound=distanceThreshold, p=2, workers=4)
                        disknn = disknn[..., -knn_n:]
                        vertex_index = vertex_index[..., -knn_n:]
                    else:
                        disknn, vertex_index = tree.query(centroid_src, k=knn_n, distance_upper_bound=distanceThreshold,
                                                          p=2)

                    knn_mask = vertex_index != n_tar  # delete the vertex whose distance upper bound

                    v_src = np.tile(np.array(i_src, dtype='str').reshape((n_src, -1)), (1, knn_n))[knn_mask]

                    v_tar = np.array(i_tar, dtype='str')[vertex_index[knn_mask]]

                    pairs = np.concatenate([pairs, np.stack((v_src, v_tar), axis=1)], axis=0)
                    disKnnList = np.concatenate([disKnnList, disknn], axis=0)
                    subgraph_names += i_src

        subgraph = globalGraph.induced_subgraph(featype_index)

        subgraph.add_edges(pairs[:, 0:2])
        multiple_edge = subgraph.es[np.where(np.array(subgraph.is_multiple()))[0].tolist()]
        subgraph.delete_edges(multiple_edge)

        subgraph_edge = subgraph.get_edge_dataframe()
        subgraph_vname = subgraph.get_vertex_dataframe()['name']

        subgraph_edge['source'] = [subgraph_vname[a] for a in subgraph_edge['source'].values]
        subgraph_edge['target'] = [subgraph_vname[a] for a in subgraph_edge['target'].values]
        subgraph_edge.insert(subgraph_edge.shape[1], 'featype', featype)

        edge_info = pd.concat([edge_info, subgraph_edge])

        print(f'---Getting DisKnn features')
        feats = getGraphDisKnnFeatures(subgraph_names, disKnnList,distanceThreshold)
        for k, v in zip(feats.keys(), feats.values()):
            if k != 'name':
                globalGraph.vs[feats['name']]['Graph_' + featype + '_' + k] = v

        print(f'---Getting GraphCenter features')
        feats = getGraphCenterFeatures(subgraph)
        for k, v in zip(feats.keys(), feats.values()):
            if k != 'name':
                globalGraph.vs[feats['name']]['Graph_' + featype + '_' + k] = v

    print(f"{'Graph features cost':#^40s}, {time.time() - t3:*^10.2f}")

    # Stroma blocker
    t4 = time.time()
    # Neopla->T, Inflam->I, Connec->S, Normal->N
    centroid_T = globalGraph.induced_subgraph(neoplaIDs).vs['Centroid']
    centroid_I = globalGraph.induced_subgraph(inflamIDs).vs['Centroid']
    centroid_S = globalGraph.induced_subgraph(connecIDs).vs['Centroid']
    Ttree = cKDTree(centroid_T)
    STree = cKDTree(centroid_S)
    dis, pairindex_T = Ttree.query(centroid_I, k=1) 
    paircentroid_T = np.array(centroid_T)[pairindex_T] 
    blocker = []
    for Tcoor, Icoor, r in tqdm(zip(centroid_I, paircentroid_T, dis), total=len(centroid_I)):
        set1 = set(STree.query_ball_point(Tcoor, r))
        set2 = set(STree.query_ball_point(Icoor, r))
        blocker.append(len(set1 & set2))
    globalGraph.vs[inflamIDs]['stromaBlocker'] = blocker
    print(f"{'stroma blocker cost':#^40s}, {time.time() - t4:*^10.2f}")

    return globalGraph, edge_info, flag
