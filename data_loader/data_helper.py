import numpy as np
import os
import pickle
import networkx as nx
from glob import glob
from os.path import basename
import copy


NUM_LABELS = {'ENZYMES': 3, 'COLLAB': 0, 'IMDBBINARY': 0, 'IMDBMULTI': 0, 'MUTAG': 7, 'NCI1': 37, 'NCI109': 38,
              'PROTEINS': 3, 'PTC': 22, 'DD': 89}
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_dataset(ds_name):
    """
    construct graphs and labels from dataset text in data folder
    :param ds_name: name of data set you want to load
    :return: two numpy arrays of shape (num_of_graphs).
            the graphs array contains in each entry a ndarray represent adjacency matrix of a graph of shape (num_vertex, num_vertex, num_vertex_labels)
            the labels array in index i represent the class of graphs[i]
    """
    
    directory = BASE_DIR + "/data/benchmark_graphs/{0}/{0}.txt".format(ds_name)
    graphs = []
    labels = []
    with open(directory, "r") as data:
        num_graphs = int(data.readline().rstrip().split(" ")[0])
        for i in range(num_graphs):
            graph_meta = data.readline().rstrip().split(" ")
            num_vertex = int(graph_meta[0])
            curr_graph = np.zeros(shape=(num_vertex, num_vertex, NUM_LABELS[ds_name]+1), dtype=np.float32)
            labels.append(int(graph_meta[1]))
            for j in range(num_vertex):
                vertex = data.readline().rstrip().split(" ")
                if NUM_LABELS[ds_name] != 0:
                    curr_graph[j, j, int(vertex[0])+1]= 1.
                for k in range(2,len(vertex)):
                    curr_graph[j, int(vertex[k]), 0] = 1.
            curr_graph = normalize_graph(curr_graph)
            graphs.append(curr_graph)
    graphs = np.array(graphs)
    for i in range(graphs.shape[0]):
        graphs[i] = np.transpose(graphs[i], [2,0,1])
        
    return graphs, np.array(labels)
    
def load_dataset_SimGNN(ds_name):
    """
    :return: graphs numpy array of shape (num_of_graphs)
            ged numpy matrix of shape (num_of_graphs, num_of_graphs)
    """
    directory = BASE_DIR + "/data/SimGNN_graphs/{}".format(ds_name)
    graphs = iterate_get_graphs(directory)
    num_graphs = len(graphs)
    
    pickle_path = directory+"/geds.pickle"
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            geds = pickle.load(f)
    else:
        geds = np.zeros((num_graphs,num_graphs))
        for i in range(num_graphs):
            for j in range(i):
                geds[i,j] = next(nx.optimize_graph_edit_distance( graphs[i], graphs[j], node_subst_cost=is_diff, edge_subst_cost=is_diff ))
                geds[j,i] = geds[i,j]
            print("Calculating GEDs {}/{}".format(i,num_graphs))
        with open(pickle_path, 'wb') as f:
            pickle.dump(geds, f)
    
    # construct dict injectively mapping node attribute dicts -> integers
    node_num_att = 0
    node_att_map = {}
    for graph in graphs:
        nodes = list(graph.nodes.data())
        for node in nodes:   # each node is (node, att_dict) tuple
            att = copy.deepcopy(node[1])
            del att['label']    # this is always unique to each node; we don't care about this
            att = str(sorted(att.items()))   # can't hash a dict, so we use this instead
            if att not in node_att_map:
                node_att_map[att] = node_num_att
                node_num_att += 1
                
    # convert graphs to numpy format
    npgraphs = np.empty(len(graphs), dtype=object)
    for i in range(len(graphs)):
        npgraph = nx_to_np(graphs[i], node_att_map)
        npgraph = np.transpose(npgraph, [1,2,0])
        npgraph = normalize_graph(npgraph)
        npgraph = np.transpose(npgraph, [2,0,1])
        npgraphs[i] = npgraph
    return npgraphs, geds
    
def iterate_get_graphs(dir):
    graphs = []
    for file in glob(dir + '/*.gexf'):
        gid = int(basename(file).split('.')[0])
        g = nx.read_gexf(file)
        # g.graph['gid'] = gid
        graphs.append(g)
        if not nx.is_connected(g):
            raise RuntimeError('{} not connected'.format(gid))
    return graphs
    
def is_diff(att1,att2):
    """
    Cost = 1 if any attribute other than "label" is different, 0 otherwise
    """
    return all(att1[key] == att2[key] or key == "label" for key in att1)
    
def nx_to_np(G, node_att_map):
    n = G.number_of_nodes()
    nodes = list(G.nodes.data())
    node_num_att = len(node_att_map)
            
    # TODO handle edge attributes (PPGN wasn't built for edge attributes)
    
    # populate node attributes
    if node_num_att == 1: # one label is no label
        npG = np.zeros((1, n, n))
        npG[0,:,:] = nx.to_numpy_matrix(G)
    else:
        npG = np.zeros((node_num_att+1, n, n))
        npG[0,:,:] = nx.to_numpy_matrix(G)
        for i in range(n):
            att = copy.deepcopy(nodes[i][1])
            del att['label']    # this is always unique to each node; we don't care about this
            att = str(sorted(att.items()))   # can't hash a dict, so we use this instead
            att = node_att_map[att]
            npG[1+att,i,i] = 1
    
    return npG


def load_qm9(target_param):
    """
    Constructs the graphs and labels of QM9 data set, already split to train, val and test sets
    :return: 6 numpy arrays:
                 train_graphs: N_train,
                 train_labels: N_train x 12, (or Nx1 is target_param is not False)
                 val_graphs: N_val,
                 val_labels: N_train x 12, (or Nx1 is target_param is not False)
                 test_graphs: N_test,
                 test_labels: N_test x 12, (or Nx1 is target_param is not False)
                 each graph of shape: 19 x Nodes x Nodes (CHW representation)
    """
    train_graphs, train_labels = load_qm9_aux('train', target_param)
    val_graphs, val_labels = load_qm9_aux('val', target_param)
    test_graphs, test_labels = load_qm9_aux('test', target_param)
    return train_graphs, train_labels, val_graphs, val_labels, test_graphs, test_labels


def load_qm9_aux(which_set, target_param):
    """
    Read and construct the graphs and labels of QM9 data set, already split to train, val and test sets
    :param which_set: 'test', 'train' or 'val'
    :param target_param: if not false, return the labels for this specific param only
    :return: graphs: (N,)
             labels: N x 12, (or Nx1 is target_param is not False)
             each graph of shape: 19 x Nodes x Nodes (CHW representation)
    """
    base_path = BASE_DIR + "/data/QM9/QM9_{}.p".format(which_set)
    graphs, labels = [], []
    with open(base_path, 'rb') as f:
        data = pickle.load(f)
        for instance in data:
            labels.append(instance['y'])
            nodes_num = instance['usable_features']['x'].shape[0]
            graph = np.empty((nodes_num, nodes_num, 19))
            for i in range(13):
                # 13 features per node - for each, create a diag matrix of it as a feature
                graph[:, :, i] = np.diag(instance['usable_features']['x'][:, i])
            graph[:, :, 13] = instance['usable_features']['distance_mat']
            graph[:, :, 14] = instance['usable_features']['affinity']
            graph[:, :, 15:] = instance['usable_features']['edge_features']  # shape n x n x 4
            graphs.append(graph)
    graphs = np.array(graphs)
    for i in range(graphs.shape[0]):
        graphs[i] = np.transpose(graphs[i], [2, 0, 1])
    labels = np.array(labels).squeeze()  # shape N x 12
    if target_param is not False:  # regression over a specific target, not all 12 elements
        labels = labels[:, target_param].reshape(-1, 1)  # shape N x 1

    return graphs, labels


def get_train_val_indexes(num_val, ds_name):
    """
    reads the indexes of a specific split to train and validation sets from data folder
    :param num_val: number of the split
    :param ds_name: name of data set
    :return: indexes of the train and test graphs
    """
    directory = BASE_DIR + "/data/benchmark_graphs/{0}/10fold_idx".format(ds_name)
    train_file = "train_idx-{0}.txt".format(num_val)
    train_idx=[]
    with open(os.path.join(directory, train_file), 'r') as file:
        for line in file:
            train_idx.append(int(line.rstrip()))
    test_file = "test_idx-{0}.txt".format(num_val)
    test_idx = []
    with open(os.path.join(directory, test_file), 'r') as file:
        for line in file:
            test_idx.append(int(line.rstrip()))
    return train_idx, test_idx


def get_parameter_split(ds_name):
    """
    reads the indexes of a specific split to train and validation sets from data folder
    :param ds_name: name of data set
    :return: indexes of the train and test graphs
    """
    directory = BASE_DIR + "/data/benchmark_graphs/{0}/".format(ds_name)
    train_file = "tests_train_split.txt"
    train_idx=[]
    with open(os.path.join(directory, train_file), 'r') as file:
        for line in file:
            train_idx.append(int(line.rstrip()))
    test_file = "tests_val_split.txt"
    test_idx = []
    with open(os.path.join(directory, test_file), 'r') as file:
        for line in file:
            test_idx.append(int(line.rstrip()))
    return train_idx, test_idx


def group_same_size(graphs, labels):
    """
    group graphs of same size to same array
    :param graphs: numpy array of shape (num_of_graphs) of numpy arrays of graphs adjacency matrix
    :param labels: numpy array of labels
    :return: two numpy arrays. graphs arrays in the shape (num of different size graphs) where each entry is a numpy array
            in the shape (number of graphs with this size, num vertex, num. vertex, num vertex labels)
            the second arrayy is labels with correspons shape
    """
    sizes = list(map(lambda t: t.shape[1], graphs))
    indexes = np.argsort(sizes)
    graphs = graphs[indexes]
    if is_ged(labels):
        labels = labels[indexes][:,indexes]
    else:
        labels = labels[indexes]
    r_graphs = []
    r_labels = []
    one_size = []
    start = 0
    size = graphs[0].shape[1]
    for i in range(len(graphs)):
        if graphs[i].shape[1] == size:
            one_size.append(np.expand_dims(graphs[i], axis=0))
        else:
            r_graphs.append(np.concatenate(one_size, axis=0))
            if is_ged(labels):
                r_labels.append(np.array(labels[start:i][:,start:i]))
            else:
                r_labels.append(np.array(labels[start:i]))
            start = i
            one_size = []
            size = graphs[i].shape[1]
            one_size.append(np.expand_dims(graphs[i], axis=0))
    r_graphs.append(np.concatenate(one_size, axis=0))
    if is_ged(labels):
        r_labels.append(np.array(labels[start:][:,start:]))
    else:
        r_labels.append(np.array(labels[start:]))
    return r_graphs, r_labels


# helper method to shuffle each same size graphs array
def shuffle_same_size(graphs, labels):
    r_graphs, r_labels = [], []
    for i in range(len(labels)):
        curr_graph, curr_labels = shuffle(graphs[i], labels[i])
        r_graphs.append(curr_graph)
        r_labels.append(curr_labels)
    return r_graphs, r_labels


def split_to_batches(graphs, labels, size):
    """
    split the same size graphs array to batches of specified size
    last batch is in size num_of_graphs_this_size % size
    :param graphs: array of arrays of same size graphs
    :param labels: the corresponding labels of the graphs
    :param size: batch size
    :return: two arrays. graphs array of arrays in size (batch, num vertex, num vertex. num vertex labels)
                corresponds labels
    """
    r_graphs = []
    r_labels = []
    for k in range(len(graphs)):
        r_graphs = r_graphs + np.split(graphs[k], [j for j in range(size, graphs[k].shape[0], size)])
        if is_ged(labels[k]):
            split = []
            for j in range( labels[k].shape[0] // size ):
                split = split + [ labels[k] [j*size : (j+1)*size] [:, j*size : (j+1)*size] ]
            if labels[k].shape[0] % size != 0:
                split = split + [ labels[k] [(labels[k].shape[0]//size)*size:] [:, (labels[k].shape[0]//size)*size:] ]
            r_labels = r_labels + split
        else:
            r_labels = r_labels + np.split(labels[k], [j for j in range(size, labels[k].shape[0], size)])

    # Avoid bug for batch_size=1, where instead of creating numpy array of objects, we had numpy array of floats with
    # different sizes - could not reshape
    ret1, ret2 = np.empty(len(r_graphs), dtype=object), np.empty(len(r_labels), dtype=object)
    ret1[:] = r_graphs
    ret2[:] = r_labels
    return ret1, ret2


# helper method to shuffle the same way graphs and labels arrays
def shuffle(graphs, labels):
    shf = np.arange(labels.shape[0], dtype=np.int32)
    np.random.shuffle(shf)
    if is_ged(labels):
        return np.array(graphs)[shf], labels[shf][:,shf]
    else:
        return np.array(graphs)[shf], labels[shf]


def normalize_graph(curr_graph):

    split = np.split(curr_graph, [1], axis=2)

    adj = np.squeeze(split[0], axis=2)
    deg = np.sqrt(np.sum(adj, 0))
    deg = np.divide(1., deg, out=np.zeros_like(deg), where=deg!=0)
    normal = np.diag(deg)
    norm_adj = np.expand_dims(np.matmul(np.matmul(normal, adj), normal), axis=2)
    ones = np.ones(shape=(curr_graph.shape[0], curr_graph.shape[1], curr_graph.shape[2]), dtype=np.float32)
    spred_adj = np.multiply(ones, norm_adj)
    labels= np.append(np.zeros(shape=(curr_graph.shape[0], curr_graph.shape[1], 1)), split[1], axis=2)
    return np.add(spred_adj, labels)

def is_ged(labels):
    return labels.shape == (labels.shape[0], labels.shape[0])

if __name__ == '__main__':
    graphs, labels = load_dataset("MUTAG")
    a, b = get_train_val_indexes(1, "MUTAG")
    print(np.transpose(graphs[a[0]], [1, 2, 0])[0])
