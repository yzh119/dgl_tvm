from tvm import relay
from collections import namedtuple
from dgl.data import load_data
from tvm.contrib import graph_runtime
import time
import tvm
import numpy as np
import networkx as nx
import argparse


def GraphConv(
            layer_name,
            input_dim,
            output_dim,
            adj,
            input,
            activation=None,
            norm=None,
            ):
    r"""
    Parameters
    ----------
    layer_name: str
    Name of layer

    input_dim: int
    Input dimension per node feature

    output_dim: int,
    Output dimension per node feature

    adj: namedtuple,
    Graph representation (Adjacency Matrix) in Sparse Format (`data`, `indices`, `indptr`),
    where `data` has shape [num_nonzeros], indices` has shape [num_nonzeros], `indptr` has shape [num_nodes + 1]

    input: relay.Expr,
    Input feature to current layer with shape [num_nodes, input_dim]

    norm: relay.Expr,
    Norm passed to this layer to normalize features before and after Convolution.

    activation: <function relay.op.nn>,
    Activation function applies to the output. e.g. relay.nn.{relu, sigmoid, log_softmax, softmax, leaky_relu}


    Returns
    ----------
    output: tvm.relay.Expr
    The Output Tensor for this layer [num_nodes, output_dim]
    """
    if norm is not None:
        input = relay.multiply(input, norm)
    weight = relay.var(layer_name + "_weight", shape=(input_dim, output_dim))
    weight_transposed = relay.transpose(weight)
    dense = relay.nn.dense(weight_transposed, input)
    output = relay.nn.sparse_dense(dense, adj)
    output_transposed = relay.transpose(output)
    if norm is not None:
        output_transposed = relay.multiply(output_transposed, norm)
    if activation is not None:
        output_transposed = activation(output_transposed)
    return output_transposed


def load_dataset(args):
    dataset = load_data(args)

    params = {}
    params['infeats'] = dataset.features.astype('float32') # Only support float32 as feature for now

    # Remove self-loops to avoid duplicate passing of a node's feature to itself
    g = dataset.graph
    g.remove_edges_from(g.selfloop_edges())
    g.add_edges_from(zip(g.nodes, g.nodes))

    # Generate adjacency matrix
    adjacency = nx.to_scipy_sparse_matrix(g)
    params['data'] = adjacency.data.astype('float32')
    params['indices'] = adjacency.indices.astype('int32')
    params['indptr'] = adjacency.indptr.astype('int32')

    # Normalization w.r.t. node degrees
    degs = [g.in_degree[i] for i in range(g.number_of_nodes())]
    params['norm'] = np.power(degs, -0.5).astype('float32')
    params['norm'] = params['norm'].reshape((params['norm'].shape[0], 1))

    return params, dataset.num_labels

argparser = argparse.ArgumentParser('TVM GraphConv')
argparser.add_argument('--num-hidden', '-l', default=1, type=int, help='Number of hidden layers')
argparser.add_argument('--hidden_dim', '-d', default=16, type=int, help='Hidden size')
argparser.add_argument('--dataset', '-data', default='cora', type=str, help='Dataset used.')
args = argparser.parse_args()

num_hidden = args.num_hidden
hidden_dim = args.hidden_dim

target = 'llvm' #'llvm -mcpu=core-avx2'
activation = relay.nn.relu
params, num_classes = load_dataset(args)

# Check shape of features
assert len(params['infeats'].shape) == 2
nnodes, input_dim = params['infeats'].shape

# Check validity of adjacency matrix
assert params['data'] is not None and params['indices'] is not None and params['indptr'] is not None
assert nnodes == params['indptr'].shape[0] - 1

layers = []

# Define input features, norms, adjacency matrix
infeats = relay.var("infeats", shape=(nnodes, input_dim))

norm = relay.Constant(tvm.nd.array(params['norm']))

data = relay.Constant(tvm.nd.array(params['data']))
indices = relay.Constant(tvm.nd.array(params['indices']))
indptr = relay.Constant(tvm.nd.array(params['indptr']))

Adjacency = namedtuple('Adjacency', ['data', 'indices', 'indptr'])
adj = Adjacency(data, indices, indptr)

# Generate Input Layer
layers.append(GraphConv(
    layer_name= 'in',
    input_dim= input_dim,
    output_dim= hidden_dim,
    adj = adj,
    input= infeats,
    activation= activation,
    norm= norm,
))

# Generate Hidden Layers
for i in range(num_hidden):
    layers.append(GraphConv(
        layer_name= str(i),
        input_dim= hidden_dim,
        output_dim= hidden_dim,
        adj = adj,
        input= layers[-1],
        activation= activation,
        norm= norm,
    ))

# Generate Output Layer
layers.append(GraphConv(
    layer_name= 'out',
    input_dim= hidden_dim,
    output_dim= num_classes,
    adj = adj,
    input= layers[-1],
    activation= activation,
    norm= norm,
))
output = layers[-1]

# Analyze free variables and generate function
func = relay.Function(relay.analysis.free_vars(output), output)

# Set up weights. You can modify this part and use your own trained weights.
params['in_weight'] = np.ones((input_dim, hidden_dim), dtype='float32')
params['out_weight'] = np.ones((hidden_dim, num_classes), dtype='float32')
for i in range(num_hidden):
    params["%s_weight"%(str(i))] = np.ones((hidden_dim, hidden_dim), dtype='float32')

# Generate graph and library
with relay.build_config(opt_level=0): # Currently only support opt_level=0
    graph, lib, params = relay.build(func, target, params=params)
    lib.save("lib.o")

# Generate module for llvm
ctx = tvm.context(target, 0)
m = graph_runtime.create(graph, lib, ctx)
m.set_input(**params)

print("finished compiling, testing inference time cost")
totaltime = 0
for i in range(100):
    st = time.time()
    # One forward pass on the entire network
    m.run()
    end = time.time()
    # Retrieve output Tensor as numpy array
    outval = m.get_output(0).asnumpy()

    totaltime += (end-st)

print("100 Cycle Average Forward Pass Time ", totaltime/100)

import os
if not os.path.exists('logs'):
    os.mkdir('logs')
with open('logs/tvm_{}'.format('_'.join('{}-{}'.format(k, v) for k, v in vars(args).items())), 'w') as f:
    print((totaltime/100) * 1000, file=f)
