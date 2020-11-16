import numpy as np
import networkx as nx
from louvain import modularity
from louvain import change_of_modularity
from louvain import visit_its_neighbors
from louvain import increase_of_modularity
from louvain import compression
from louvain import create_the_dictionary_of_coummunity
from louvain import zerolize_the_be_visited
from louvain import fusion
from louvain import find_the_key_of_index
from louvain import list_of_the_empty_community_number
from louvain import clean_the_empty_line_and_colone
from louvain import order_the_dictionary
from louvain import modifiy_the_comunity


def test_liste_of_the_empty_community_number():
    A = {0: [0], 1: [1], 2: [2], 3: [3], 4: [4], 5: [5], 6: [6]}
    assert list_of_the_empty_community_number(A) == []
    B = {0: [], 1: [1], 2: [2], 3: [3], 4: [4], 5: [5], 6: [6]}
    assert list_of_the_empty_community_number(B) == [0]
    C = {0: [], 1: [1], 2: [], 3: [], 4: [], 5: [5], 6: [6]}
    assert list_of_the_empty_community_number(C) == [0, 2, 3, 4]
    D = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
    assert list_of_the_empty_community_number(D) == [0, 1, 2, 3, 4, 5, 6]


def test_dictionary():
    G = nx.Graph()
    G.add_edge(0, 1, weight=1)
    G.add_edge(0, 2, weight=1)
    G.add_edge(1, 2, weight=1)
    G.add_edge(2, 3, weight=1)
    G.add_edge(3, 4, weight=1)
    G.add_edge(4, 5, weight=1)
    G.add_edge(5, 6, weight=1)
    G.add_edge(3, 6, weight=1)
    A = nx.to_numpy_matrix(G)
    assert create_the_dictionary_of_coummunity(A) == {0: [0], 1: [1], 2: [2], 3: [3], 4: [4], 5: [5], 6: [6]}


def test_moudularity_after_compression():
    G = nx.Graph()
    G.add_edge(0, 1, weight=1)
    G.add_edge(0, 2, weight=1)
    G.add_edge(1, 2, weight=1)
    G.add_edge(2, 3, weight=1)
    G.add_edge(3, 4, weight=1)
    G.add_edge(4, 5, weight=1)
    G.add_edge(5, 6, weight=1)
    G.add_edge(3, 6, weight=1)
    A = nx.to_numpy_matrix(G)
    B = compression(A, 0, 1)
    assert modularity(B) == -0.0546875


def test_transformation_graph_to_matrix():
    graph = nx.Graph()
    graph.add_edge(0, 1)
    matrix = nx.to_numpy_matrix(graph)
    np.testing.assert_array_equal(matrix, np.array([[0, 1], [1, 0]]))


def test_modularity_graph_a_node():
    matrix_graph = np.array([[0]])
    assert 0 == modularity(matrix_graph)


def test_modularity_graph_plus_grand():
    matrix_graph = np.array([[0, 1], [1, 0]])
    matrix_graph1 = np.array([[0, 0, 0], [0, 4, 0], [0, 0, 1]])
    matrix_graph2 = np.array([[4, 0], [0, 1]])
    matrix_graph3 = np.array([[0, 0,2],[0,0,0],[2,0,0]])
    matrix_graph4 = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
    assert -0.5 == modularity(matrix_graph)
    assert -0.5 == modularity(matrix_graph3)
    assert modularity(matrix_graph1) == modularity(matrix_graph2)
    assert -0.375==modularity(matrix_graph4)

def test_moudularity_with_seven_nodes():
    G = nx.Graph()
    G.add_edge(0, 1, weight=1)
    G.add_edge(0, 2, weight=1)
    G.add_edge(1, 2, weight=1)
    G.add_edge(2, 3, weight=1)
    G.add_edge(3, 4, weight=1)
    G.add_edge(4, 5, weight=1)
    G.add_edge(5, 6, weight=1)
    G.add_edge(3, 6, weight=1)
    A = nx.to_numpy_matrix(G)
    assert -0.1484375 == modularity(A)


def test_visit_its_neighbors():
    G = nx.Graph()
    G.add_edge(0, 1, weight=1, com=1)
    G.add_edge(0, 2, weight=1, com=2)
    G.add_edge(1, 2, weight=1, com=3)
    G.add_edge(2, 3, weight=1, com=4)
    G.add_edge(3, 4, weight=1, com=5)
    G.add_edge(4, 5, weight=1, com=6)
    G.add_edge(5, 6, weight=1, com=7)
    G.add_edge(3, 6, weight=1, com=8)
    # A = nx.to_numpy_matrix(G)
    assert [0, 2] == visit_its_neighbors(G, 1)


def test_increase_of_modularity():
    G = nx.Graph()
    G.add_edge(0, 1, weight=1)
    G.add_edge(0, 2, weight=1)
    G.add_edge(1, 2, weight=1)
    G.add_edge(2, 3, weight=1)
    G.add_edge(3, 4, weight=1)
    G.add_edge(4, 5, weight=1)
    G.add_edge(5, 6, weight=1)
    G.add_edge(3, 6, weight=1)
    dic = {0: [0], 1: [1], 2: [2], 3: [3], 4: [4], 5: [5], 6: [6]}
    assert 1 == increase_of_modularity(G, 0, dic)


def test_change_of_modularity():
    B = np.array([(0, 1, 0, 1), (1, 0, 1, 0.), (0, 1, 0, 1), (1, 0, 1, 0)])
    assert 0.125 == change_of_modularity(B, 3, 2)


def test_modularity():
    assert modularity(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])) == -0.375
    assert modularity(np.array([[2, 1], [1, 0]])) == -0.125
    assert modularity(np.array(
        [[0, 1, 1, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0, 0], [1, 1, 0, 1, 0, 0, 0], [0, 0, 1, 0, 1, 0, 1],
         [0, 0, 0, 1, 0, 1, 0], [0, 0, 0, 0, 1, 0, 1], [0, 0, 0, 1, 0, 1, 0]])) == -0.1484375


# 有问题
def test_zerolization_order_problem():  # compression函数会改变matrix
    G = nx.Graph()
    G.add_edge(0, 1, weight=1)
    G.add_edge(0, 2, weight=1)
    G.add_edge(1, 2, weight=1)
    G.add_edge(2, 3, weight=1)
    G.add_edge(3, 4, weight=1)
    G.add_edge(4, 5, weight=1)
    G.add_edge(5, 6, weight=1)
    G.add_edge(3, 6, weight=1)
    A = nx.to_numpy_matrix(G)
    B = A.copy()
    D = compression(A, 0, 1)
    E = zerolize_the_be_visited(B, 0, 1)
    assert modularity(E) == modularity(D)


def test_zerolization():
    G = nx.Graph()
    G.add_edge(0, 1, weight=1)
    G.add_edge(0, 2, weight=1)
    G.add_edge(1, 2, weight=1)
    G.add_edge(2, 3, weight=1)
    G.add_edge(3, 4, weight=1)
    G.add_edge(4, 5, weight=1)
    G.add_edge(5, 6, weight=1)
    G.add_edge(3, 6, weight=1)
    A = nx.to_numpy_matrix(G)
    B = A.copy()
    E = zerolize_the_be_visited(A, 0, 1)
    D = compression(B, 0, 1)
    assert modularity(E) == modularity(D)


# 下一次注意increase_of_modularity():函数里用的len（）
# 当选中了节点以后如何目前想的办法是用cut_the_size函数将所访问列变0
# 目前考虑用字典来描述社团的关系
# 2020/1/14 裁剪为0函数不正确需要修改
def test_fusion():
    A = {0: [0], 1: [1], 2: [2], 3: [3], 4: [4], 5: [5], 6: [6]}
    assert fusion(A, 0, 1) == {0: [], 1: [1, 0], 2: [2], 3: [3], 4: [4], 5: [5], 6: [6]}
    assert fusion(A, 2, 1) == {0: [], 1: [1, 0, 2], 2: [], 3: [3], 4: [4], 5: [5], 6: [6]}
    assert fusion(A, 1, 0) == {0: [1, 0, 2], 1: [], 2: [], 3: [3], 4: [4], 5: [5], 6: [6]}
    assert fusion(A, 0, 3) == {0: [], 1: [], 2: [], 3: [3, 1, 0, 2], 4: [4], 5: [5], 6: [6]}
    assert fusion(A, 6, 3) == {0: [], 1: [], 2: [], 3: [3, 1, 0, 2, 6], 4: [4], 5: [5], 6: []}


def test_create_the_dictionary_of_coummunity():
    G = nx.Graph()
    G.add_edge(0, 1, weight=1)
    G.add_edge(0, 2, weight=1)
    G.add_edge(1, 2, weight=1)
    G.add_edge(2, 3, weight=1)
    G.add_edge(3, 4, weight=1)
    G.add_edge(4, 5, weight=1)
    G.add_edge(5, 6, weight=1)
    G.add_edge(3, 6, weight=1)
    A = nx.to_numpy_matrix(G)
    assert create_the_dictionary_of_coummunity(A) == {0: [0], 1: [1], 2: [2], 3: [3], 4: [4], 5: [5], 6: [6]}


def test_find_the_key_of_index():
    a = {0: [0], 1: [1], 2: [2], 3: [3], 4: [4], 5: [5, 6], 6: [7]}
    assert find_the_key_of_index(a, 6) == 5
    assert find_the_key_of_index(a, 7) == 6
    assert find_the_key_of_index(a, []) == None


def test_zerolization():
    A = np.eye(3)
    assert np.all(zerolize_the_be_visited(A, 0, 1) == np.array([[0, 0, 0], [0, 2, 0], [0, 0, 1]]))


def test_clean_the_empty_line_and_colone():
    B = np.array([[0, 0, 0, 0, 0, 0, 0],
                  [0, 6, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 2, 2, 0],
                  [0, 0, 0, 0, 2, 2, 0],
                  [0, 0, 0, 0, 0, 0, 0]])
    list = {0: [], 1: [1, 0, 2], 2: [], 3: [], 4: [4, 3], 5: [5, 6], 6: []}
    assert np.all(np.array([[6, 1, 0], [1, 2, 2], [0, 2, 2]]) == clean_the_empty_line_and_colone(B,list_of_the_empty_community_number(list)))

def test_order_the_dictionary():
    a = {0: [], 1: [1, 0, 2], 2: [], 3: [], 4: [4, 3], 5: [5, 6], 6: []}
    b = a.copy()
    assert order_the_dictionary(b) == {0: [1, 0, 2], 1: [4, 3], 2: [5, 6]}
    a = {0: [], 1: [], 2: [], 3: [], 4: [4, 3], 5: [5, 6], 6: []}
    b = a.copy()
    assert order_the_dictionary(b) == {0: [4, 3], 1: [5, 6]}


def test_modifiy_the_comunity():
    a = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}
    b = {0: [1, 0, 2], 1: [4, 3], 2: [5, 6]}
    assert modifiy_the_comunity(b, a) == {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
    a = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}
    b = {0: [1, 0, 2, 4], 1: [3], 2: [5, 6]}
    assert modifiy_the_comunity(b, a) == {0: 0, 1: 0, 2: 0, 3: 1, 4: 0, 5: 2, 6: 2}
