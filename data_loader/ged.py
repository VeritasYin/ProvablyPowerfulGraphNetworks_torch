import numpy as np
import networkx as nx

def ged(G1, G2):
    """
    :param G1, G2: graphs whose distance is computed; assumed shape 
    :(num_vertex_labels+1, num_vertex, num_vertex)
    """
    nxG1 = np_to_nx(G1)
    nxG2 = np_to_nx(G2)
    return nx.graph_edit_distance(nxG1, nxG2, node_subst_cost=is_diff)
    
def is_diff(n1,n2):
    """
    node_subst_cost function. Cost = 1 if any attribute is different, 0 otherwise
    """
    return n1 != n2
    
def np_to_nx(G):
    """
    assumed shape (num_vertex_labels+1, num_vertex, num_vertex)
    """
    num_vertex = G.shape[1]
    num_vertex_labels = G.shape[0]-1
    print(G[0,:,:].shape)
    nxG = nx.Graph(G[0,:,:])
    for j in range(num_vertex_labels):
        nx.set_node_attributes(nxG, j, 0)
        for i in range(num_vertex):
            if G[j+1,i,i] == 1:
                nxG.nodes[i][j] = 1
    return nxG
