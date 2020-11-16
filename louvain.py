import numpy as np
import networkx as nx

def modularity(matrix):
    m = np.sum(matrix)
    if m==0:
        return 0
    sum_communaute = 0
    for i in range( matrix.shape[0]):
        sum_communaute += matrix[i, i] / m - (np.sum(matrix[i]) / m) **2
    return sum_communaute

def change_of_modularity(matrix, begin, neighbor_of_begin):
    if neighbor_of_begin != None:
        change = (matrix[begin, neighbor_of_begin] - np.sum(matrix[begin]) * np.sum(matrix[neighbor_of_begin]) / matrix.sum()) / (matrix.sum()/ 2)
        return change#计算begin和begin邻居合并的变化
    else:
        print(begin,'没有邻居',neighbor_of_begin)
        return 0

#calculer les augementation de moudularite entre n et les voisins de n,et retourner le neoud qui est plus haut
def increase_of_modularity(graph, begin,dic):#返回变化值列表
    list_de_changement=[]
    Transformation_of_Matrix = nx.to_numpy_matrix(graph)
    list_of_node=visit_its_neighbors(graph, begin)#list of node中是begin的邻居
    if begin in list_of_node:
        list_of_node.remove(begin)
    #print(begin,'邻居节点是:',list_of_node)
    for i in list_of_node:
        list_de_changement.append(change_of_modularity(Transformation_of_Matrix, begin, find_the_key_of_index(dic,i)))
    #print('改变的模块度',list_de_changement)
    if (list_de_changement!=[] and max(list_de_changement)>=0):#M变化值是正数时才有意义
        #print('最高改变 ', list_of_node[list_de_changement.index(max(list_de_changement))])
        return list_of_node[list_de_changement.index(max(list_de_changement))]
    else:
        return None

def find_the_key_of_index(dic,inde):
    if inde !=[]:
        for i in dic:
            if inde in dic[i]:
                return i
    return None

def list_of_the_empty_community_number(dic):
    list = []
    for i in dic:
        if dic[i]==[]:
            list.append(i)
    return list

def visit_its_neighbors(graph,begin):
    list_of_nodes=[]
    for neighbors in graph[begin]:
        list_of_nodes.append(neighbors)
    #print ('Le voisin de',begin,'est',list_of_nodes)
    return list_of_nodes

#ajouter les noeus qui est deja visite
def ajouter_dans_visited(a,visited):
    visited.append(a)
    return visited

def compression(matrix, b, c):
    matrix[:, b]= matrix[:, b] + matrix[:, c]
    matrix[b]= matrix[b] + matrix[c]
    matrix = np.delete(matrix, c, axis=0)
    matrix = np.delete(matrix, c, axis=1)
    return matrix

#输入矩阵，形成字典
def create_the_dictionary_of_coummunity(matrix):
    dic = dict.fromkeys(range(matrix.shape[1]), [0])
    for x in range(matrix.shape[1]):
        dic[x] = [x]
    return dic

def create_the_unchange(matrix):
    dic = dict.fromkeys(range(matrix.shape[1]), [0])
    for x in range(matrix.shape[1]):
        dic[x] = x
    return dic

def cut_the_size(matrix,be_added,comunity_number):
    matrix[:,comunity_number]=matrix[:,comunity_number]+matrix[:,be_added]
    matrix[comunity_number]=matrix[comunity_number]+matrix[be_added]
    matrix[be_added] = 0
    matrix[:, be_added] = 0
    return matrix

#重命名快捷方式shift+f6
def zerolize_the_be_visited(matrix,begin,be_add):#将出发节点的下一个加入出发节点的社区,be_add为零
    matrix[:,be_add]=matrix[:,begin]+matrix[:,be_add]
    matrix[be_add]=matrix[begin]+matrix[be_add]
    matrix[begin]=0
    matrix[:,begin]=0
    return matrix

#融合函数会改变原始字典
def fusion(dictionairy, community_number,community_neignber_number):
    if community_neignber_number!=None:
        if (dictionairy[community_neignber_number]!=[]):
            dictionairy[community_neignber_number]=dictionairy[community_neignber_number]+dictionairy[community_number]
        else:
            dictionairy[community_neignber_number]=dictionairy[community_number]
        dictionairy[community_number] = []
        return dictionairy
    else:
        return dictionairy

def clean_the_empty_line_and_colone(matrix,list_of_empty):
    for number in reversed(list_of_empty):
        matrix = np.delete(matrix, number, axis=0)
        matrix = np.delete(matrix, number, axis=1)
    return matrix

def order_the_dictionary(dict):#使用前要深拷贝%
    dict=dict.copy()
    for i in list(dict.keys()):
        if len(dict[i])==0:
            del dict[i]
    ret={}
    for up,i in enumerate(list(dict.keys()),0):
        ret[up]=dict[i]
    return ret

def modifiy_the_comunity(dict,dict_unchange):
    dict_unchange_c=dict_unchange.copy()
    for i in dict_unchange_c:
        new_number=find_the_key_of_index(dict,dict_unchange_c[i])
        dict_unchange_c[i]=new_number
    return dict_unchange_c

def bd_first_search(g, root):
    edges_list = None
    edges_list = list(nx.traversal.bfs_edges(g, root))
    # 整理结果
    nodes_list = None
    nodes_list = list(edges_list[0])
    for k, v in edges_list[1:]:
        if v not in nodes_list:
            nodes_list.append(v)
    return nodes_list
