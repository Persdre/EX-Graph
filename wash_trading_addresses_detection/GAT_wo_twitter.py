#!/usr/bin/env python
# coding: utf-8

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import random
import pickle as pkl
import copy
from dgl.nn.pytorch import GATConv
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score


# Read data
def load_graph_data(filename):
    with open(filename, 'rb') as f:
        return pkl.load(f)


G_train_dgl = load_graph_data('G_train_dgl.gpickle')
G_val_dgl = load_graph_data('G_val_dgl.gpickle')
G_test_dgl = load_graph_data('G_test_dgl.gpickle')


class GATModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, dropout_rate=0.1):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, num_heads=num_heads)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, num_heads)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.conv3 = GATConv(hidden_dim * num_heads, out_dim, num_heads)

    def forward(self, g, h):
        h = self.conv1(g, h).flatten(1)
        h = F.elu(self.dropout1(h))
        h = self.conv2(g, h).flatten(1)
        h = F.elu(self.dropout2(h))
        h = self.conv3(g, h).mean(1)
        return h


def prepare_data(g, feature_name='ethereum_features'):
    labels = g.ndata['label'].squeeze()
    zero_indices = torch.where(labels == 0)[0]
    one_indices = torch.where(labels == 1)[0]
    min_count = min(zero_indices.shape[0], one_indices.shape[0])
    selected_zero_indices = zero_indices[torch.randperm(zero_indices.shape[0])[:min_count]]
    selected_one_indices = one_indices[torch.randperm(one_indices.shape[0])[:min_count]]
    selected_indices = torch.cat((selected_zero_indices, selected_one_indices))
    selected_indices = selected_indices[torch.randperm(selected_indices.shape[0])]
    subgraph = dgl.node_subgraph(g, selected_indices)
    return subgraph, subgraph.ndata[feature_name], subgraph.ndata['label'].squeeze()


def main():
    in_feats = G_train_dgl.ndata['ethereum_features'].shape[1]
    hidden_size = 128
    out_feats = 2
    num_heads = 3

    # Set random seed
    seed = random.randint(0, 1000)
    print(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    for i in range(10):
        model = GATModel(in_feats, hidden_size, out_feats, num_heads)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()
        best_val_loss = float('inf')
        best_model = None
        num_epochs = 200

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            subgraph, selected_features, selected_labels = prepare_data(G_train_dgl)
            logits = model(subgraph, selected_features)
            labels = F.one_hot(selected_labels, num_classes=out_feats).float()
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                subgraph, selected_features, selected_labels = prepare_data(G_val_dgl)
                logits = model(subgraph, selected_features)
                labels = F.one_hot(selected_labels, num_classes=out_feats).float()
                val_loss = criterion(logits, labels)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(model)
                torch.save(model.state_dict(), 'best_model.pt')

            print(f"Epoch: {epoch + 1}/{num_epochs}, Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}")

        best_model.eval()
        with torch.no_grad():
            subgraph, selected_features, ground_truth = prepare_data(G_test_dgl)
            logits = best_model(subgraph, selected_features)
            predicted_probs = F.softmax(logits, dim=1)[:, 1]
            predicted_labels = (predicted_probs > 0.5).float()

            metrics = {
                "AUC": roc_auc_score(ground_truth.detach().numpy(), predicted_probs.detach().numpy()),
                "F1": f1_score(ground_truth.detach().numpy(), predicted_labels.detach().numpy()),
                "Precision": precision_score(ground_truth.detach().numpy(), predicted_labels.detach().numpy()),
                "Recall": recall_score(ground_truth.detach().numpy(), predicted_labels.detach().numpy()),
                "Accuracy": accuracy_score(ground_truth.detach().numpy(), predicted_labels.detach().numpy()),
                "Macro-F1": f1_score(ground_truth.detach().numpy(), predicted_labels.detach().numpy(), average='macro'),
                "Macro-Precision": precision_score(ground_truth.detach().numpy(), predicted_labels.detach().numpy(), average='macro'),
                "Macro-recall": recall_score(ground_truth.detach().numpy(), predicted_labels.detach().numpy(), average='macro')
            }

            with open("GAT_wo_results.txt", "a") as f:
                f.write(f"Random seed: {seed}, Epoch: {epoch + 1}/{num_epochs}, Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}, "
                        + ", ".join([f"{key}: {value:.4f}" for key, value in metrics.items()]) + "\n")

            print(", ".join([f"{key}: {value:.4f}" for key, value in metrics.items()]) + "\n")


if __name__ == '__main__':
    main()
