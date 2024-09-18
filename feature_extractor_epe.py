
import networkx as nx
import numpy as np
import argparse
import os
import sent2vec
import pickle
import glob
import re
from multiprocessing import Pool
from functools import partial
from pdg.points_get import get_CVPC_node_list, get_param_node, get_pointers_node, get_all_array, get_all_sensitiveAPI, get_all_integeroverflow_point, get_all_control_structure
from pdg.slicer_op import slice_nodes_to_single_subgraph
from pdg.node import Node
from pdg.edge import Edge

def parse_options():
    parser = argparse.ArgumentParser(description='Image-based Vulnerability Detection.')
    parser.add_argument('-i', '--input', help='The path of a dir which consists of some dot_files')
    parser.add_argument('-o', '--out', help='The path of output.', required=True)
    parser.add_argument('-m', '--model', help='The path of model.', required=True)
    args = parser.parse_args()
    return args


def graph_extraction(dot):
    graph = nx.drawing.nx_pydot.read_dot(dot)
    return graph


def sentence_embedding(sentence):
    emb = sent2vec_model.embed_sentence(sentence)
    return emb[0]


def split_code(code):
    # 在特殊符号前后添加空格
    code = re.sub(r'([{}();,\[\]])', r' \1 ', code)
    # 在运算符前后添加空格
    code = re.sub(r'(\+|-|\*|/|=|==|!=|<=|>=|&&|\|\||!|&|\||<<|>>)', r' \1 ', code)
    # 处理数组初始化中的花括号
    code = re.sub(r'\{', r' { ', code)
    code = re.sub(r'\}', r' } ', code)
    # 处理模板中的尖括号
    code = re.sub(r'(<|>)', r' \1 ', code)
    return code


def get_dependency_flow_feature(graph):
    flow_edges = []
    num_nodes = graph.number_of_nodes()
    node_list = list(graph.nodes())
    
    if num_nodes == 0 or len(node_list) == 0:
        # print("graph is empty!")
        return flow_edges
    else:
        # print("graph nodes: ", graph.nodes(data=True))
        for u, v, attrs in graph.edges(data=True):
            # 获取节点的代码
            # print("attrs: ", graph.nodes[u])
            u_code = graph.nodes[u]["code"]
            v_code = graph.nodes[v]["code"]
            u_vec = sentence_embedding(split_code(u_code))
            v_vec = sentence_embedding(split_code(v_code))
            u_vec = np.array(u_vec)
            v_vec = np.array(v_vec)
            # 将节点的代码向量作为流边的特征
            flow_edge_vec = np.concatenate((u_vec, v_vec), axis=0)
            # print(f"Edge ({u}, {v}):", u_code, v_code)
            flow_edges.append(flow_edge_vec)
            
        return flow_edges


def extract_cdg_ddg(pdg):
    """
    从程序依赖图（PDG）中提取控制依赖图（CDG）和数据依赖图（DDG）。
    参数:
    pdg: nx.DiGraph - 程序依赖图。
    返回:
    cdg: nx.DiGraph - 控制依赖图。
    ddg: nx.DiGraph - 数据依赖图。
    """
    cdg = nx.DiGraph()
    ddg = nx.DiGraph()
    # 添加边并根据边类型添加相应节点
    for u, v, attrs in pdg.edges(data=True):
        edge_type = attrs.get("label", "")
        if "CDG" in edge_type:
            if not cdg.has_node(u):
                cdg.add_node(u, **pdg.nodes[u])
            if not cdg.has_node(v):
                cdg.add_node(v, **pdg.nodes[v])
            
            cdg.add_edge(u, v, **attrs)
        elif "DDG" in edge_type:
            if not ddg.has_node(u):
                ddg.add_node(u, **pdg.nodes[u])
            if not ddg.has_node(v):
                ddg.add_node(v, **pdg.nodes[v])

            ddg.add_edge(u, v, **attrs)
    return ddg, cdg


def redefine_pdg(pdg):
    # 所有边
    edges = [Edge(u, v, attrs) for u, v, attrs in pdg.edges(data=True)]
    # 所有节点
    node_dict = {node: Node(node, attrs, edges) for node, attrs in pdg.nodes(data=True)}
    # generate the new pdg
    new_pdg = nx.DiGraph()
    for node, attrs in node_dict.items():
        node_data = node_dict[node]
        new_pdg.add_node(node, type=node_data.node_type, label=node_data.label, code=node_data.code)
    
    for edge in edges:
        new_pdg.add_edge(edge.node_in, edge.node_out, label=edge.edge_type)
        
    return new_pdg


def pdg_slice(pdg):
    '''
    对PDG进行切片
    '''
    # 所有边
    edges = [Edge(u, v, attrs) for u, v, attrs in pdg.edges(data=True)]
    # 所有节点
    node_dict = {node: Node(node, attrs, edges) for node, attrs in pdg.nodes(data=True)}
    # print('get_CVPC_node_list!')
    cvpcs = get_CVPC_node_list(node_dict)
    # print('slice!')
    vsdg = slice_nodes_to_single_subgraph(edges, cvpcs, node_dict)
    # print('slice done!')
    return vsdg
    

def graph_feature_extraction(G):
    '''
    # 获取特征
    :param G:
    :return:
    '''
    try:
        in_degree_dict = dict(G.in_degree())
        out_degree_dict = dict(G.out_degree())
        degree_dict = dict(G.degree())
        degree_cen_dict = nx.degree_centrality(G)
        closeness_cen_dict = nx.closeness_centrality(G)
        katz_cen_dict = nx.katz_centrality(G)
        betweenness_cen_dict = nx.betweenness_centrality(G)
        graph_feature = []
        for node, attrs in G.nodes(data=True):
            code = attrs["code"]
            line_vec = sentence_embedding(split_code(code))
            line_vec = np.array(line_vec) 

            attr_encoding = np.array([attrs['label'],
                                            in_degree_dict[node],
                                            out_degree_dict[node],
                                            degree_dict[node],
                                            degree_cen_dict[node], 
                                            closeness_cen_dict[node], 
                                            katz_cen_dict[node],
                                            betweenness_cen_dict[node]])
            
            line_feature = np.concatenate((line_vec, attr_encoding), axis=0)
            graph_feature.append(line_feature)
            
        return graph_feature
    except:
        return None
    

def get_feature(dot):
    try:
        # print(dot)
        pdg = graph_extraction(dot)
        if pdg.number_of_nodes() == 0 or len(pdg.nodes()) == 0:
            return None
        # print("start slice")
        sliced_pdg = pdg_slice(pdg)
        # 如果切片后的图节点数为0，则使用未切片的图
        if sliced_pdg.number_of_nodes() == 0 or len(sliced_pdg.nodes()) == 0:
            sliced_pdg = redefine_pdg(pdg)

        pdg_feature = graph_feature_extraction(sliced_pdg)
        ddg, cdg = extract_cdg_ddg(sliced_pdg)
        ddg_flow_edges_featrue = get_dependency_flow_feature(ddg)
        cdg_flow_edges_featrue = get_dependency_flow_feature(cdg)
        return pdg_feature, ddg_flow_edges_featrue, cdg_flow_edges_featrue
    except:
        print("#######error:", dot, "failed!")
        return None


def store_in_pkl(dot, out, existing_files):
    dot_name = dot.split('/')[-1].split('.dot')[0]
    if dot_name in existing_files:
        # print(dot_name + " already exists!")
        return None
    else:
        # print(dot_name)
        multi_features = get_feature(dot)
        if multi_features == None or multi_features[0] == None:
            return None
        else:
            # print(dot_name)
            pdg_feature, ddg_feature, cdg_feature = multi_features
            out_pkl = out + dot_name + '.pkl'
            data = [pdg_feature, [ddg_feature, cdg_feature]]
            with open(out_pkl, 'wb') as f:
                pickle.dump(data, f)
                
        # print(dot_name + " done!")


def main():
    args = parse_options()
    dir_name = args.input
    out_path = args.out
    trained_model_path = args.model
    print("dir_name: ", dir_name)
    print("out_path: ", out_path)
    print("trained_model_path: ", trained_model_path)
    global sent2vec_model
    sent2vec_model = sent2vec.Sent2vecModel()
    sent2vec_model.load_model(trained_model_path)

    if dir_name[-1] == '/':
        dir_name = dir_name
    else:
        dir_name += "/"

    dotfiles = glob.glob(dir_name + '*.dot')
    print("dotfiles: ", len(dotfiles))

    if out_path[-1] == '/':
        out_path = out_path
    else:
        out_path += '/'

    if not os.path.exists(out_path):
        os.makedirs(out_path)
        
    existing_files = glob.glob(out_path + "/*.pkl")
    existing_files = [f.split('/')[-1].split('.pkl')[0] for f in existing_files]
    # print("existing_files: ", len(existing_files))
    # print(existing_files[:10])
    
    pool = Pool(10)
    pool.map(partial(store_in_pkl, out=out_path, existing_files=existing_files), dotfiles)


    sent2vec_model.release_shared_mem(trained_model_path)

import time

if __name__ == '__main__':
    print("begin to extract feature...")
    ts = time.time()
    main() 
    print("TIme cost: ", time.time() - ts)
    print("feature extraction done!")