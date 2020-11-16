import networkx as nx
import numpy as np
import copy
import matplotlib.pyplot as plt
from louvain import *

# The node number should always continue
G = nx.Graph()

file = open("data.txt")
read_in = np.loadtxt('data.txt')

for i in read_in:
    G.add_edge(i[0],i[1])

matrix_de_graph = nx.to_numpy_matrix(G)
D = G.copy()
G = nx.from_numpy_matrix(matrix_de_graph)
matrix_next = matrix_de_graph.copy()
counter = 0
dic_unchange = create_the_unchange(matrix_next)  # 索引对应的是社区号
#print('La modularite au debut',modularity(matrix_de_graph))
while (modularity(matrix_de_graph) <= modularity(matrix_next)):
    # bfs，需要注意一定要用bfs，在数量很小的时候差别不大，但数量很大的时候，dfs会导致分配不到最大值
    counter = counter + 1
    nodes=bd_first_search(G,root = 0)
    dic = create_the_dictionary_of_coummunity(matrix_next)  # {0: [0], 1: [1], 2: [2], 3: [3], 4: [4], 5: [5], 6: [6]}
    # print('字典',dic)
    visited = []
    dic_support = dic.copy()
    for i in nodes:  # 在DFS中遍历
        if i not in visited:  # 如果还没访问就继续
            ajouter_dans_visited(i, visited)  # 加入被访问list
            if increase_of_modularity(G, i, dic) != None:  # 当还有正增长时
                biggest_neighbor = increase_of_modularity(G, i, dic)  # 用biggest去代替
                ajouter_dans_visited(biggest_neighbor, visited)  # 把其加入已访问
                matrix_compressed = zerolize_the_be_visited(matrix_next, find_the_key_of_index(dic, i),find_the_key_of_index(dic, biggest_neighbor))  # 压缩后变0
                #print('matrix compress',matrix_compressed,'the key of index i',find_the_key_of_index(dic, i),'the key of index biggest',find_the_key_of_index(dic, biggest_neighbor))
                fusion(dic, i, find_the_key_of_index(dic, biggest_neighbor))  # 将i融合进社区
            else:
                matrix_compressed=matrix_next
                continue
    if counter == 1:
        final_dict = dic_unchange

    pure_dic = order_the_dictionary(dic)
    final_dict = modifiy_the_comunity(pure_dic, final_dict)
    matrix_next = clean_the_empty_line_and_colone(matrix_compressed, list_of_the_empty_community_number(dic))

    G = nx.from_numpy_matrix(matrix_next)


    if dic == dic_support:
        break

ncolor = list(final_dict.values())
nx.draw(D, with_labels=True, node_color=ncolor, node_size=3000, cmap=plt.cm.Reds)
plt.show()



