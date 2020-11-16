import numpy as np
from louvain import modularity
import networkx as nx

G = nx.Graph()
G.add_edge(0, 1)
G.add_edge(1, 2)
G.add_edge(0, 2)
G.add_edge(2, 3)
G.add_edge(3, 9)
G.add_edge(3, 8)
G.add_edge(3, 7)
G.add_edge(8, 9)
G.add_edge(8, 7)
G.add_edge(3, 6)
G.add_edge(3, 4)
G.add_edge(4, 5)
G.add_edge(6, 5)

def bd_first_search(g, root):

    edges_list = None
    edges_list = list(nx.traversal.bfs_edges(g, root))


    # 整理结果
    nodes_list = None
    nodes_list = list(edges_list[0])
    for k, v in edges_list[1:]:
        # 可以不判断k值，定在nodes_list中
        if v not in nodes_list:
            nodes_list.append(v)

    return nodes_list

print(bd_first_search(G,0))