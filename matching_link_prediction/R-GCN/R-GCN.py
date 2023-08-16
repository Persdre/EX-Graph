import time
import math
import torch
import argparse

import utils
from model import *

import numpy as np
import torch.nn.functional as F
np.random.seed(1)

class BaseRGCN(nn.Module):
    def __init__(self, node_attri, num_nodes, h_dim, out_dim, num_rels, num_bases,
                 num_hidden_layers=1, dropout=0,
                 use_self_loop=False, use_cuda=False):
        super(BaseRGCN, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.use_cuda = use_cuda

        # create rgcn layers
        self.build_model(node_attri)

    def build_model(self, node_attri):
        self.layers = nn.ModuleList()
        # i2h
        i2h = self.build_input_layer(node_attri)
        if i2h is not None:
            self.layers.append(i2h)
        # h2h
        for idx in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)
        # h2o
        h2o = self.build_output_layer()
        if h2o is not None:
            self.layers.append(h2o)

    def build_input_layer(self, node_attri):
        return None

    def build_hidden_layer(self, idx):
        raise NotImplementedError

    def build_output_layer(self):
        return None

    def forward(self, g, h, r, norm):
        for layer in self.layers:
            h = layer(g, h, r, norm)
        return h
    
    
class EmbeddingLayer(nn.Module):
    def __init__(self, num_nodes, h_dim):
        super(EmbeddingLayer, self).__init__()
        self.embedding = torch.nn.Embedding(num_nodes, h_dim)

    def forward(self, g, h, r, norm):
        return self.embedding(h.squeeze())
    
    
class EmbeddingLayerAttri(nn.Module):
    def __init__(self, node_attri):
        super(EmbeddingLayerAttri, self).__init__()
        self.embedding = torch.nn.Embedding.from_pretrained(torch.from_numpy(node_attri))

    def forward(self, g, h, r, norm):
        return self.embedding(h.squeeze())
    
    
class RGCN(BaseRGCN):
    def build_input_layer(self, node_attri):
        if node_attri is not None:
            return EmbeddingLayerAttri(node_attri)
        return EmbeddingLayer(self.num_nodes, self.h_dim)

    def build_hidden_layer(self, idx):
        act = F.relu if idx < self.num_hidden_layers - 1 else None
        if idx==0:
            return RelGraphConv(self.h_dim, self.out_dim, self.num_rels, "basis", self.num_bases, activation=act, self_loop=True, dropout=self.dropout)
        return RelGraphConv(self.out_dim, self.out_dim, self.num_rels, "basis",
                self.num_bases, activation=act, self_loop=True,
                dropout=self.dropout)
    

class TrainModel(nn.Module):
    def __init__(self, node_attri, num_nodes, o_dim, num_rels, nlabel, num_bases=-1,
                 num_hidden_layers=1, dropout=0, use_cuda=False, reg_param=0):
        super(TrainModel, self).__init__()
        
        if node_attri is None:
            self.rgcn = RGCN(node_attri, num_nodes, o_dim, o_dim, num_rels * 2, num_bases, num_hidden_layers, dropout, use_cuda)
        else:            
            self.rgcn = RGCN(node_attri, num_nodes, node_attri.shape[1], o_dim, num_rels * 2, num_bases, num_hidden_layers, dropout, use_cuda)
        self.reg_param = reg_param
        
        if nlabel==0:
            self.supervised = False
            self.w_relation = nn.Parameter(torch.Tensor(num_rels, o_dim))
            nn.init.xavier_uniform_(self.w_relation, gain=nn.init.calculate_gain('relu'))
        else:
            self.supervised = True
            self.LinearLayer = torch.nn.Linear(o_dim, nlabel)

    def calc_score(self, embedding, triplets):
        # DistMult
        s = embedding[triplets[:,0]]
        r = self.w_relation[triplets[:,1]]
        o = embedding[triplets[:,2]]
        score = torch.sum(s * r * o, dim=1)
        return score

    def forward(self, g, h, r, norm):
        output = self.rgcn.forward(g, h, r, norm)
        if self.supervised:
            pred = self.LinearLayer(output)
        else:
            pred = None
        return output, pred

    def unsupervised_regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))

    def get_unsupervised_loss(self, g, embed, triplets, labels):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        score = self.calc_score(embed, triplets)
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        reg_loss = self.unsupervised_regularization_loss(embed)
        return predict_loss + self.reg_param * reg_loss
    
    def supervised_regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2))
    
    def get_supervised_loss(self, embed, matched_labels, matched_index, multi):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        if multi: 
            predict_loss = F.binary_cross_entropy(torch.sigmoid(embed[matched_index]), matched_labels)
        else:
            predict_loss = F.nll_loss(F.log_softmax(embed[matched_index]), matched_labels)
        reg_loss = self.supervised_regularization_loss(embed)
        return predict_loss + self.reg_param * reg_loss
    
    


import dgl
import torch
import numpy as np
from collections import defaultdict


def load_supervised(args, link, node, train_pool):
    
    num_nodes, num_rels, train_data = 0, 0, []
    train_indices = defaultdict(list)
    with open(link, 'r') as file:
        for index, line in enumerate(file):
            if index==0:
                num_nodes, num_rels = line[:-1].split(' ')
                num_nodes, num_rels = int(num_nodes), int(num_rels)
                print(f'#nodes: {num_nodes}, #relations: {num_rels}')
            else:
                line = np.array(line[:-1].split(' ')).astype(int)
                train_data.append(line)                
                if line[0] in train_pool:
                    train_indices[line[0]].append(index-1)                    
                if line[-1] in train_pool:
                    train_indices[line[-1]].append(index-1)
    
    if args.attributed=='True':
        node_attri = {}
        with open(node, 'r') as file:
            for line in file:
                line = line[:-1].split('\t')
                node_attri[int(line[0])] = np.array(line[1].split(',')).astype(np.float32)
        return np.array(train_data), num_nodes, num_rels, train_indices, len(train_indices), np.array([node_attri[k] for k in range(len(node_attri))]).astype(np.float32)
    elif args.attributed=='False':    
        return np.array(train_data), num_nodes, num_rels, train_indices, len(train_indices), None


def load_label(train_label):
    
    train_pool, train_labels, all_labels, multi = set(), {}, set(), False
    with open(train_label, 'r') as file:
        for line in file:
            node, label = line[:-1].split('\t')
            node = int(node)
            train_pool.add(node)
            if multi or ',' in label:
                multi = True
                label = np.array(label.split(',')).astype(int)
                for each in label:
                    all_labels.add(label)
                train_labels[node] = label
            else:
                label = int(label)
                train_labels[node] = label
                all_labels.add(label)
    
    return train_pool, train_labels, len(all_labels), multi


def load_unsupervised(args, link, node):
    
    num_nodes, num_rels, train_data = 0, 0, []
    with open(link, 'r') as file:
        for index, line in enumerate(file):
            if index==0:
                num_nodes, num_rels = line[:-1].split(' ')
                num_nodes, num_rels = int(num_nodes), int(num_rels)
                print(f'#nodes: {num_nodes}, #relations: {num_rels}')
            else:
                line = np.array(line[:-1].split(' ')).astype(int)
                train_data.append(line)
    
    if args.attributed=='True':
        node_attri = {}
        with open(node, 'r') as file:
            for line in file:
                line = line[:-1].split('\t')
                node_attri[int(line[0])] = np.array(line[1].split(',')).astype(np.float32)
        return np.array(train_data), num_nodes, num_rels, np.array([node_attri[k] for k in range(len(node_attri))]).astype(np.float32)
    elif args.attributed=='False':
        return np.array(train_data), num_nodes, num_rels, None


def save(args, embs):
    
    with open(f'{args.output}', 'w') as file:
        file.write(f'size={args.n_hidden}, negative={args.negative_sample}, lr={args.lr}, dropout={args.dropout}, regularization={args.regularization}, grad_norm={args.grad_norm}, num_bases={args.n_bases}, num_layers={args.n_layers}, num_epochs={args.n_epochs}, graph_batch_size={args.graph_batch_size}, graph_split_size={args.graph_split_size}, edge_sampler={args.edge_sampler}, supervised={args.supervised}, attributed={args.attributed}\n')
        for index, emb in enumerate(embs):
            file.write(f'{index}\t')
            file.write(' '.join(emb.astype(str)))
            file.write('\n')
    
    return


#######################################################################
#
# Utility function for building training and testing graphs
#
#######################################################################

def get_adj_and_degrees(num_nodes, triplets):
    """ Get adjacency list and degrees of the graph
    """
    degrees = np.zeros(num_nodes).astype(int)
    for i,triplet in enumerate(triplets):
        degrees[triplet[0]] += 1
        degrees[triplet[2]] += 1

    return degrees

def sample_edge_uniform(degrees, n_triplets, sample_size):
    """Sample edges uniformly from all the edges."""
    all_edges = np.arange(n_triplets)
    return np.random.choice(all_edges, sample_size, replace=False)

def add_labeled_edges(edges, train_indices, ntrain, if_train, label_batch_size, batch_index=0):
    
    if if_train:
        sampled_index = set(np.random.choice(np.arange(ntrain), label_batch_size, replace=False))
    else:
        sampled_index = set(np.arange(batch_index*label_batch_size, min(ntrain, (batch_index+1)*label_batch_size)))
        
    new_edges, sampled_nodes = [], set()
    for index, (labeled_node, node_edges) in enumerate(train_indices.items()):
        if index in sampled_index:
            sampled_nodes.add(labeled_node)
            new_edges.append(np.array(node_edges))
    new_edges = np.concatenate(new_edges)
    new_edges = np.unique(np.concatenate([edges, new_edges]))
    
    return new_edges, sampled_nodes

def correct_order(node_id, sampled_nodes, train_labels, multi, nlabel):
    
    matched_labels, matched_index = [], []
    for index, each in enumerate(node_id):
        if each in sampled_nodes:
            if multi: 
                curr_label = np.zeros(nlabel).astype(int)
                curr_label[train_labels[each]] = 1
                matched_labels.append(curr_label)
            else: 
                matched_labels.append(train_labels[each])
            matched_index.append(index)
    
    return np.array(matched_labels), np.array(matched_index)

def generate_sampled_graph_and_labels_supervised(triplets, sample_size, split_size, num_rels, degrees, negative_rate, sampler, train_indices, train_labels, multi, nlabel, ntrain, if_train=True, label_batch_size=512, batch_index=0):
    """Get training graph and signals
    First perform edge neighborhood sampling on graph, then perform negative
    sampling to generate negative samples
    """
    # perform edge neighbor sampling
    if sampler == "uniform":
        edges = sample_edge_uniform(degrees, len(triplets), sample_size)
    else:
        raise ValueError("Sampler type must be either 'uniform' or 'neighbor'.")
        
    edges, sampled_nodes = add_labeled_edges(edges, train_indices, ntrain, if_train, label_batch_size, batch_index)

    # relabel nodes to have consecutive node ids
    edges = triplets[edges]
    src, rel, dst = edges.transpose()
    uniq_v, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))
    relabeled_edges = np.stack((src, rel, dst)).transpose()

    matched_labels, matched_index = correct_order(uniq_v, sampled_nodes, train_labels, multi, nlabel)
    
    # negative sampling
#     samples, labels = negative_sampling(relabeled_edges, len(uniq_v),
#                                         negative_rate)

    # further split graph, only half of the edges will be used as graph
    # structure, while the rest half is used as unseen positive samples
    split_size = int(sample_size * split_size)
    graph_split_ids = np.random.choice(np.arange(sample_size),
                                       size=split_size, replace=False)
    src = src[graph_split_ids]
    dst = dst[graph_split_ids]
    rel = rel[graph_split_ids]

    # build DGL graph
#     print("# sampled nodes: {}, # sampled edges: {}".format(len(uniq_v), len(src)*2))
    g, rel, norm = build_graph_from_triplets(len(uniq_v), num_rels, (src, rel, dst))
    return g, uniq_v, rel, norm, matched_labels, matched_index


def generate_sampled_graph_and_labels_unsupervised(triplets, sample_size, split_size,
                                      num_rels, degrees,
                                      negative_rate, sampler="uniform"):
    """Get training graph and signals
    First perform edge neighborhood sampling on graph, then perform negative
    sampling to generate negative samples
    """
    # perform edge neighbor sampling
    if sampler == "uniform":
        edges = sample_edge_uniform(degrees, len(triplets), sample_size)
    else:
        raise ValueError("Sampler type must be either 'uniform' or 'neighbor'.")

    # relabel nodes to have consecutive node ids
    edges = triplets[edges]
    src, rel, dst = edges.transpose()
    uniq_v, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))
    relabeled_edges = np.stack((src, rel, dst)).transpose()

    # negative sampling
    samples, labels = negative_sampling(relabeled_edges, len(uniq_v),
                                        negative_rate)

    # further split graph, only half of the edges will be used as graph
    # structure, while the rest half is used as unseen positive samples
    split_size = int(sample_size * split_size)
    graph_split_ids = np.random.choice(np.arange(sample_size),
                                       size=split_size, replace=False)
    src = src[graph_split_ids]
    dst = dst[graph_split_ids]
    rel = rel[graph_split_ids]

    # build DGL graph
#     print("# sampled nodes: {}, # sampled edges: {}".format(len(uniq_v), len(src)*2))
    g, rel, norm = build_graph_from_triplets(len(uniq_v), num_rels, (src, rel, dst))
    return g, uniq_v, rel, norm, samples, labels


def comp_deg_norm(g):
    g = g.local_var()
    in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    norm = 1.0 / in_deg
    norm[np.isinf(norm)] = 0
    return norm

def build_graph_from_triplets(num_nodes, num_rels, triplets):
    """ Create a DGL graph. The graph is bidirectional because RGCN authors
        use reversed relations.
        This function also generates edge type and normalization factor
        (reciprocal of node incoming degree)
    """
    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    src, rel, dst = triplets
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    rel = np.concatenate((rel, rel + num_rels))
    edges = sorted(zip(dst, src, rel))
    dst, src, rel = np.array(edges).transpose()
    g.add_edges(src, dst)
    norm = comp_deg_norm(g)
    return g, rel, norm

def negative_sampling(pos_samples, num_entity, negative_rate):
    size_of_batch = len(pos_samples)
    num_to_generate = size_of_batch * negative_rate
    neg_samples = np.tile(pos_samples, (negative_rate, 1))
    labels = np.zeros(size_of_batch * (negative_rate + 1), dtype=np.float32)
    labels[: size_of_batch] = 1
    values = np.random.randint(num_entity, size=num_to_generate)
    choices = np.random.uniform(size=num_to_generate)
    subj = choices > 0.5
    obj = choices <= 0.5
    neg_samples[subj, 0] = values[subj]
    neg_samples[obj, 2] = values[obj]

    return np.concatenate((pos_samples, neg_samples)), labels

def node_norm_to_edge_norm(g, node_norm):
    g = g.local_var()
    # convert to edge norm
    g.ndata['norm'] = node_norm
    g.apply_edges(lambda edges : {'norm' : edges.dst['norm']})
    return g.edata['norm']

def main(args):
    
    # load graph data
    print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) + 'start loading...', flush=True)
    if args.supervised=='True':
        train_pool, train_labels, nlabels, multi = utils.load_label(args.label)
        train_data, num_nodes, num_rels, train_indices, ntrain, node_attri = utils.load_supervised(args, args.link, args.node, train_pool)
    elif args.supervised=='False':
        train_data, num_nodes, num_rels, node_attri = utils.load_unsupervised(args, args.link, args.node)
        nlabels = 0
    print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) + 'finish loading...', flush=True)
    
    # check cuda 
    use_cuda=False
    
    if use_cuda:
        torch.cuda.set_device(args.gpu)
    print('check 1', flush=True)
    # create model
    model = TrainModel(node_attri, num_nodes,
                args.n_hidden,
                num_rels, nlabels,
                num_bases=args.n_bases,
                num_hidden_layers=args.n_layers,
                dropout=args.dropout,
                use_cuda=use_cuda,
                reg_param=args.regularization)
    print('check 2', flush=True)
    if use_cuda:
        model.cuda()
    print('check 3', flush=True)
    # build adj list and calculate degrees for sampling
    degrees = utils.get_adj_and_degrees(num_nodes, train_data)
    print('check 4', flush=True)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # training loop
    print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) + "start training...", flush=True)
    for epoch in range(args.n_epochs):
        model.train()

        # perform edge neighborhood sampling to generate training graph and data
        if args.supervised=='True':
            g, node_id, edge_type, node_norm, matched_labels, matched_index = \
            utils.generate_sampled_graph_and_labels_supervised(
                train_data, args.graph_batch_size, args.graph_split_size,
                num_rels, degrees, args.negative_sample, args.edge_sampler, 
                train_indices, train_labels, multi, nlabels, ntrain, if_train=True, label_batch_size=args.label_batch_size)
            if multi: matched_labels = torch.from_numpy(matched_labels).float()
            else: matched_labels = torch.from_numpy(matched_labels).long()
        elif args.supervised=='False':        
            g, node_id, edge_type, node_norm, data, labels = \
            utils.generate_sampled_graph_and_labels_unsupervised(
                train_data, args.graph_batch_size, args.graph_split_size,
                num_rels, degrees, args.negative_sample,
                args.edge_sampler)
            data, labels = torch.from_numpy(data), torch.from_numpy(labels)

        # set node/edge feature
        node_id = torch.from_numpy(node_id).view(-1, 1).long()
        edge_type = torch.from_numpy(edge_type)
        edge_norm = node_norm_to_edge_norm(g, torch.from_numpy(node_norm).view(-1, 1))
        
        deg = g.in_degrees(range(g.number_of_nodes())).float().view(-1, 1)
        if use_cuda:
            node_id, deg, g = node_id.cuda(), deg.cuda(), g.to('cuda')
            edge_type, edge_norm = edge_type.cuda(), edge_norm.cuda()
            if args.supervised=='True': matched_labels = matched_labels.cuda()
            elif args.supervised=='False': data, labels = data.cuda(), labels.cuda()

        embed, pred = model(g, node_id, edge_type, edge_norm)
        if args.supervised=='True': loss = model.get_supervised_loss(pred, matched_labels, matched_index, multi)
        elif args.supervised=='False': loss = model.get_unsupervised_loss(g, embed, data, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm) # clip gradients
        optimizer.step()
        optimizer.zero_grad()  
        
        print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) + 
              "Epoch {:05d} | Loss {:.4f}".format(epoch, loss.item()), flush=True)      

    print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) + "training done", flush=True)
    
    print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) + "start output...", flush=True)
    model.eval()
    if args.attributed=='True':
        np.random.shuffle(train_data)
        node_emb, node_over = np.zeros((num_nodes, args.n_hidden)), set()
        batch_total = math.ceil(len(train_data)/args.graph_batch_size)
        for batch_num in range(batch_total):

            # perform edge neighborhood sampling to generate training graph and data
            g, old_node_id, edge_type, node_norm, data, labels = \
                utils.generate_sampled_graph_and_labels_unsupervised(
                    train_data, args.graph_batch_size, args.graph_split_size,
                    num_rels, degrees, args.negative_sample,
                    args.edge_sampler)

            # set node/edge feature
            node_id = torch.from_numpy(old_node_id).view(-1, 1).long()
            edge_type = torch.from_numpy(edge_type)
            edge_norm = node_norm_to_edge_norm(g, torch.from_numpy(node_norm).view(-1, 1))
            if use_cuda:
                node_id, g = node_id.cuda(), g.to('cuda')
                edge_type, edge_norm = edge_type.cuda(), edge_norm.cuda()

            embed, _ = model(g, node_id, edge_type, edge_norm)
            node_emb[old_node_id] = embed.detach().cpu().numpy().astype(np.float32)   
        
            for each in old_node_id:
                node_over.add(each)
        
            print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) + 
                  f'finish output batch nubmer {batch_num} -> {batch_total}', flush=True)

        utils.save(args, node_emb)
        
    elif args.attributed=='False':
        utils.save(args, model.rgcn.layers[0].embedding.weight.detach().cpu().numpy())
    
    return
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--link", type=str, required=True,
            help="dataset to use")
    parser.add_argument("--node", type=str, required=True,
            help="dataset to use")
    parser.add_argument("--label", type=str, required=True,
            help="dataset to use")
    parser.add_argument('--output', required=True, type=str, 
            help='Output embedding file')
    parser.add_argument("--dropout", type=float, default=0.2,
            help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=50,
            help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-bases", type=int, default=100,
            help="number of weight bases for each relation")
    parser.add_argument("--n-layers", type=int, default=2,
            help="number of propagation rounds")
    parser.add_argument("--n-epochs", type=int, default=2000,
            help="number of minimum training epochs")    
    parser.add_argument("--regularization", type=float, default=0.01,
            help="regularization weight")
    parser.add_argument("--grad-norm", type=float, default=1.0,
            help="norm to clip gradient to")
    parser.add_argument("--label-batch-size", type=int, default=512)
    parser.add_argument("--graph-batch-size", type=int, default=200000,
            help="number of edges to sample in each iteration")
    parser.add_argument("--graph-split-size", type=float, default=0.5,
            help="portion of edges used as positive sample")
    parser.add_argument("--negative-sample", type=int, default=5,
            help="number of negative samples per positive sample")
    parser.add_argument("--edge-sampler", type=str, default="uniform",
            help="type of edge sampler: 'uniform' or 'neighbor'")
    parser.add_argument("--attributed", type=str, default="False")
    parser.add_argument("--supervised", type=str, default="False")

    args = parser.parse_args()
    print(args, flush=True)
    main(args)