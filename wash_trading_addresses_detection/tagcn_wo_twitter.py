#!/usr/bin/env python
# coding: utf-8

import argparse
import copy
import dgl
import dgl.function as fn
import numpy as np
import pickle as pkl
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import TAGConv
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from tqdm import trange


class TAGCN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super(TAGCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(TAGConv(in_feats, n_hidden, activation=activation))
        for _ in range(n_layers - 1):
            self.layers.append(TAGConv(n_hidden, n_hidden, activation=activation))
        self.layers.append(TAGConv(n_hidden, n_classes))
        self.dropout = dropout

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h


def create_balanced_subgraph(graph, label_key='label', feature_key='ethereum_features'):
    labels = graph.ndata[label_key].squeeze()
    zero_indices = torch.where(labels == 0)[0]
    one_indices = torch.where(labels == 1)[0]
    min_count = min(zero_indices.shape[0], one_indices.shape[0])
    selected_zero_indices = zero_indices[torch.randperm(zero_indices.shape[0])[:min_count]]
    selected_one_indices = one_indices[torch.randperm(one_indices.shape[0])[:min_count]]
    selected_indices = torch.cat((selected_zero_indices, selected_one_indices))
    selected_indices = selected_indices[torch.randperm(selected_indices.shape[0])]
    subgraph = dgl.node_subgraph(graph, selected_indices)
    return subgraph


# Load datasets
with open('G_train_dgl.gpickle', 'rb') as f:
    G_train_dgl = pkl.load(f)
with open('G_val_dgl.gpickle', 'rb') as f:
    G_val_dgl = pkl.load(f)
with open('G_test_dgl.gpickle', 'rb') as f:
    G_test_dgl = pkl.load(f)

# Get the number of input features
in_feats = G_train_dgl.ndata['ethereum_features'].shape[1]



def main():
    for i in trange(10):
        # Set seeds
        seed = random.randint(0, 1000)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        model = TAGCN(in_feats, n_hidden=128, n_classes=2, n_layers=1, activation=F.relu, dropout=0.1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()
        best_val_loss = float('inf')
        num_epochs = 200

        for epoch in trange(num_epochs):
            model.train()
            optimizer.zero_grad()

            subgraph = create_balanced_subgraph(G_train_dgl)
            selected_features = subgraph.ndata['ethereum_features'].float()
            selected_labels = subgraph.ndata['label'].squeeze()
            logits = model(subgraph, selected_features)
            labels = F.one_hot(selected_labels, num_classes=2).float()
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                subgraph = create_balanced_subgraph(G_val_dgl)
                selected_features = subgraph.ndata['ethereum_features']
                selected_labels = subgraph.ndata['label'].squeeze()
                logits = model(subgraph, selected_features)
                labels = F.one_hot(selected_labels, num_classes=2).float()
                val_loss = criterion(logits, labels)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_model.pt')

        # Evaluation
        best_model = TAGCN(in_feats, n_hidden=128, n_classes=2, n_layers=1, activation=F.relu, dropout=0.1)
        best_model.load_state_dict(torch.load('best_model.pt'))
        best_model.eval()
        with torch.no_grad():
            subgraph = create_balanced_subgraph(G_test_dgl)
            selected_features = subgraph.ndata['ethereum_features']
            ground_truth = subgraph.ndata['label'].squeeze()
            logits = best_model(subgraph, selected_features)
            predicted_probs = F.softmax(logits, dim=1)[:, 1]
            predicted_labels = (predicted_probs > 0.5).float()
            auc = roc_auc_score(ground_truth.detach().numpy(), predicted_probs.detach().numpy())
            f1 = f1_score(ground_truth.detach().numpy(), predicted_labels.detach().numpy())
            precision = precision_score(ground_truth.detach().numpy(), predicted_labels.detach().numpy())
            recall = recall_score(ground_truth.detach().numpy(), predicted_labels.detach().numpy())
            accuracy = accuracy_score(ground_truth.detach().numpy(), predicted_labels.detach().numpy())
            macro_f1 = f1_score(ground_truth.detach().numpy(), predicted_labels.detach().numpy(), average='macro')

            print(f"Epoch {i+1}, AUC: {auc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}, Macro F1: {macro_f1:.4f}")


if __name__ == '__main__':
    main()