import igraph as ig
from igraph import Graph
import pandas as pd
import os
import numpy as np
from scipy.stats import skew, kurtosis
from collections import defaultdict
from functools import partial
from tqdm import tqdm
from multiprocessing import Pool
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

def return_value(args,values):
    if args == 'min':
        return np.min(values)
    elif args == 'max':
        return np.max(values)
    elif args == 'mean':
        return np.mean(values)
    elif args == 'std':
        return np.std(values)
    elif args == 'skewness':
        value = skew(values)
        if np.isnan(value):
            return 0
        else:
            return value
    elif args == 'kurt':
        value = kurtosis(values)
        if np.isnan(value):
            return 0
        else:
            return value
 
# ref:https://python.igraph.org/en/stable/api/igraph.Graph.html
def return_graph_feature(args,graph):
    if args == 'Diameter':# Calculates the diameter of the graph.
        return graph.diameter(directed=False, unconn=True, weights=None)
    elif args == 'ECount':# Counts the number of edges.
        return graph.ecount()
    elif args == 'VCount':# Counts the number of vertices.
        return graph.vcount()
    elif args == 'EdgeConnectivity':
        '''
        Calculates the edge connectivity of the graph or between some vertices.
        The edge connectivity between two given vertices is the number of edges that have to be 
        removed in order to disconnect the two vertices into two separate components. This is 
        also the number of edge disjoint directed paths between the vertices. The edge connectivity
        of the graph is the minimal edge connectivity over all vertex pairs.
        This method calculates the edge connectivity of a given vertex pair if both the source and 
        target vertices are given. If none of them is given (or they are both negative), the overall
        edge connectivity is returned.
        '''
        return graph.edge_connectivity()
    elif args == 'CliqueNumber':
        '''
        Returns the clique number of the graph.
        The clique number of the graph is the size of the largest clique.
        '''
        return graph.clique_number()
    elif args == 'Density':
        return graph.density()
    elif args == 'AvgTransitivity':
        '''
        Calculates the average of the vertex transitivities of the graph.
        In the unweighted case, the transitivity measures the probability that two 
        neighbors of a vertex are connected. In case of the average local transitivity,
        this probability is calculated for each vertex and then the average is taken. 
        Vertices with less than two neighbors require special treatment, they will either
        be left out from the calculation or they will be considered as having zero 
        transitivity, depending on the mode parameter.
        '''
        return graph.transitivity_avglocal_undirected(mode='zero', weights=None)

    elif args == 'AssortativityDegree':
        '''
        Returns the assortativity of a graph based on vertex degrees.
        '''
        value = graph.assortativity_degree(directed=False)
        if np.isnan(value):
            return 0
        else:
            return value
    


def Pipeline(slide_id,feature_root,output_root=None):
    '''
    feature_root : root of sc_MTOP results saved
    output_root : root of graph feature saved, default:feature_root
    '''
    if not output_root:
        output_root = feature_root
    else:
        os.makedirs(output_root,exist_ok=True)
    os.makedirs(os.path.join(output_root,slide_id),exist_ok=True)
    print('Processing slide_id:{}'.format(slide_id))
    total_feat_csv = defaultdict(list)
    edge_csv = pd.read_csv(os.path.join(feature_root,slide_id,slide_id+'_'+'Edges.csv'))
    global_graph = Graph.DataFrame(edges = edge_csv ,use_vids=False,directed=False)
    verclu = Graph.community_multilevel(global_graph)
    statistics = ['min','max','mean','std','skewness','kurt']
    graph_feat_names = ['Diameter','ECount','VCount','EdgeConnectivity','CliqueNumber','Density',\
                        'AvgTransitivity','AssortativityDegree'
                        ]
    for i in tqdm(range(len(verclu))):
        subgraph = verclu.subgraph(i)
        vernames = subgraph.vs['name']
        flag = False
        feat_csvs = []
        for feat in ['T','I','S']:
            feat_csv = pd.read_csv(os.path.join(feature_root,slide_id,slide_id+'_Feats_'+feat+'.csv'))
            feat_csv = feat_csv[feat_csv['name'].isin(vernames)]
            if len(feat_csv) == 0:
                flag = True
                break
            feat_csv = feat_csv.drop(['name', 'Bbox', 'Centroid', 'CellType'],axis=1)
            feat_csvs.append(feat_csv)
        if not flag:
            total_feat_csv['cluid'].append(i)
            for feat,feat_csv in zip(['T','I','S'],feat_csvs):
                for key in feat_csv.keys():
                    for statistic in statistics:
                        new_key = key + '_' + statistic+'_'+feat
                        total_feat_csv[new_key].append(return_value(statistic, np.array(feat_csv[key].tolist())))
            for graph_feat in graph_feat_names:
                total_feat_csv[graph_feat].append(return_graph_feature(graph_feat,subgraph))
    total_feat_data = pd.DataFrame(total_feat_csv)
    total_feat_data.to_csv(os.path.join(output_root,slide_id,'{}_Graph_Feat.csv'.format(slide_id)),index=False)
    print('{} success!'.format(slide_id))


    
if __name__ == '__main__':

    feature_root = "" # Path to save cellcluster feat
    output_root = None
    pipeline = partial(Pipeline, feature_root=feature_root,output_root=output_root)
    args = os.listdir(feature_root)
    cpu_worker_num = 10
    with Pool(cpu_worker_num) as p:
        p.map(pipeline, args)

