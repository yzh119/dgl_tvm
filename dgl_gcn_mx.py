from dgl.data import load_data
from mxnet import nd, gluon
import mxnet as mx
import mxnet.gluon.nn as nn
import time
import dgl
import dgl.nn.mxnet.conv as conv
import argparse

argparser = argparse.ArgumentParser('DGL GraphConv')
argparser.add_argument('--num-hidden', '-l', default=1, type=int, help='Number of hidden layers')
argparser.add_argument('--hidden_dim', '-d', default=16, type=int, help='Hidden size')
argparser.add_argument('--dataset', '-data', default='cora', type=str, help='Dataset used.')
args = argparser.parse_args()

activation = nd.relu
data = load_data(args)

g = data.graph
g.remove_edges_from(g.selfloop_edges())
g.add_edges_from(zip(g.nodes, g.nodes))
g = dgl.DGLGraph(g)
print(g)

features = nd.array(data.features)
num_hidden = args.num_hidden
input_dim = features.shape[1]
hidden_dim = args.hidden_dim
num_classes = data.num_labels

layers = nn.Sequential() 
layers.add(conv.GraphConv(
    input_dim,
    hidden_dim,
    activation=activation
))
for _ in range(num_hidden):
    layers.add(conv.GraphConv(
        hidden_dim,
        hidden_dim,
        activation=activation
    ))
layers.add(conv.GraphConv(
    hidden_dim,
    num_classes,
    activation=activation
))
layers.initialize(ctx=mx.cpu(0))

print('testing inference time cost')
tot_time = 0
for i in range(100):
    tic = time.time()
    feat = features
    for layer in layers:
        feat = layer(feat, g)
    toc = time.time()
    tot_time += toc-tic

print('100 Cycle Average Forward Pass Time ', tot_time/100)

import os
if not os.path.exists('logs'):
    os.mkdir('logs')
with open('logs/dglmx_{}'.format('_'.join('{}-{}'.format(k, v) for k, v in vars(args).items())), 'w') as f:
    print((tot_time/100) * 1000, file=f)
