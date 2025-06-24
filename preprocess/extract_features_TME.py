import json
import os
from utils.graph_patch import constructGraphFromDict
from collections import defaultdict
import numpy as np
from utils.xml import get_windows
import pandas as pd
from multiprocessing import Pool

def fun3(args):
    json_path, patch_path, output_path = args[0], args[1], args[2]
    distanceThreshold = 100
    level = 0
    k = 5

    sample_name = os.path.basename(patch_path).rsplit('.', 1)[0]
    with open(json_path) as fp:
        print(f"{'Loading json':*^30s}")
        nucleusInfo = json.load(fp)

    globalgraph, edge_info, flag = constructGraphFromDict(patch_path, nucleusInfo, distanceThreshold, k, level)
    if flag == False:
        print(f"{sample_name}:Null data in centroid_src or centroid_tar")
    else:
        vertex_dataframe = globalgraph.get_vertex_dataframe()
        centroid = np.array(vertex_dataframe['Centroid'].tolist())

        col_dist = defaultdict(list)
        ## Neopla->T, Inflam->I, Connec->S, Normal->N
        cellType = ['T', 'I', 'S']
        for featname in vertex_dataframe.columns.values:
            if 'Graph' not in featname:
                # public feature, including cell information, Morph feature and GLCM feature
                for cell in cellType:
                    col_dist[cell] += [featname] if featname != 'Contour' else []
            else:
                # Graph feature, format like 'Graph_T-I_Nsubgraph'
                for cell in cellType:
                    featype = featname.split('_')[1]  # Graph feature type like 'T-T', 'T-I'
                    col_dist[cell] += [featname] if cell in featype else []
        cellType_save = {'T': [1],  # Neopla
                        'I': [2],  # Inflam
                        'S': [3],  # Connec
                        'N': [5]}  # Normal

        output_path = os.path.join(output_path, sample_name)
        try:
            os.makedirs(output_path)
        except:
            pass
        for i in col_dist.keys():
            vertex_csvfile = os.path.join(output_path, sample_name + '_Feats_' + i + '.csv')
            save_index = vertex_dataframe['CellType'].isin(cellType_save[i]).values
            vertex_dataframe.iloc[save_index].to_csv(vertex_csvfile, index=False, columns=col_dist[i])
        edge_csvfile = os.path.join(output_path, sample_name + '_Edges.csv')
        edge_info.to_csv(edge_csvfile, index=False)


if __name__ == '__main__':

    size = 1120
    slide_ext = '.svs'

    hovernet_root = f'/dataset/{size}/hovernet'
    slide_paths = f'/dataset/{size}/patch_png'
    output_path = f'/dataset/{size}/hovernet_sc_mtop'

    args = []
    cpu_worker_num = 10
    for slide_path in sorted(os.listdir(slide_paths)):
        slide_path = slide_path.replace(slide_ext, "")
        for patch in sorted(os.listdir(os.path.join(slide_paths, slide_path))):
                    a = os.path.join(output_path,slide_path,patch.replace('.png',''))
                    if os.path.exists(a): 
                        print(f"Directory {a} already exists, skipping.")
                        continue

                    output_path2 = os.path.join(output_path,slide_path)
                    json_path = os.path.join(hovernet_root,slide_path,'json',patch.replace('png','json'))
                    args.append((json_path, patch, output_path2))
                    # break

        # break
        
    with Pool(cpu_worker_num) as p:
        p.map(fun3, args)
    
