#!/usr/bin/env python
# coding: utf-8

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATv2Conv
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, accuracy_score
)
import numpy as np
import random
import pickle as pkl
import copy


class GATv2(nn.Module):
    def __init__(self, num_layers, in_dim, num_hidden, num_classes, heads, activation, feat_drop, attn_drop, negative_slope, residual):
        super(GATv2, self).__init__()
        self.num_layers = num_layers
        self.gatv2_layers = nn.ModuleList()
        self.activation = activation

        # Input projection (no residual)
        self.gatv2_layers.append(GATv2Conv(in_dim, num_hidden, heads[0], feat_drop, attn_drop, negative_slope, False, self.activation, bias=False, share_weights=True))

        # Hidden layers
        for l in range(1, num_layers):
            # Due to multi-head, the in_dim = num_hidden * num_heads
            self.gatv2_layers.append(GATv2Conv(num_hidden * heads[l - 1], num_hidden, heads[l], feat_drop, attn_drop, negative_slope, residual, self.activation, bias=False, share_weights=True))

        # Output projection
        self.gatv2_layers.append(GATv2Conv(num_hidden * heads[-2], num_classes, heads[-1], feat_drop, attn_drop, negative_slope, residual, None, bias=False, share_weights=True))

    def forward(self, g, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gatv2_layers[l](g, h).flatten(1)
        logits = self.gatv2_layers[-1](g, h).mean(1)
        return logits


def load_data():
    with open('G_train_dgl_twitter.gpickle', 'rb') as f:
        G_train = pkl.load(f)
    with open('G_val_dgl_twitter.gpickle', 'rb') as f:
        G_val = pkl.load(f)
    with open('G_test_dgl_twitter.gpickle', 'rb') as f:
        G_test = pkl.load(f)

    return G_train, G_val, G_test


def create_balanced_subgraph(graph, label_key):
    labels = graph.ndata[label_key].squeeze()
    zero_indices = torch.where(labels == 0)[0]
    one_indices = torch.where(labels == 1)[0]
    min_count = min(zero_indices.shape[0], one_indices.shape[0])
    selected_zero_indices = zero_indices[torch.randperm(zero_indices.shape[0])[:min_count]]
    selected_one_indices = one_indices[torch.randperm(one_indices.shape[0])[:min_count]]
    selected_indices = torch.cat((selected_zero_indices, selected_one_indices))
    selected_indices = selected_indices[torch.randperm(selected_indices.shape[0])]
    return dgl.node_subgraph(graph, selected_indices)


def main():
    G_train, G_val, G_test = load_data()
    in_feats = G_train.ndata['ethereum_twitter_combined_features'].shape[1]

    # Define model hyperparameters
    num_layers = 3
    in_dim = in_feats
    num_hidden = 128
    num_classes = 2
    heads = [3, 3, 3]
    activation = F.elu
    feat_drop = 0.1
    attn_drop = 0.1
    negative_slope = 0.2
    residual = False

    for _ in range(10):
        seed = random.randint(0, 1000)
        print(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        model = GATv2(num_layers, in_dim, num_hidden, num_classes, heads, activation, feat_drop, attn_drop, 0.1, True)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()
        best_val_loss = float('inf')
        best_model = None
        num_epochs = 200
        patience = 20

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()

            subgraph_train = create_balanced_subgraph(G_train, 'label')
            features = subgraph_train.ndata['ethereum_twitter_combined_features']
            labels = subgraph_train.ndata['label'].squeeze()
            logits = model(subgraph_train, features.float())
            labels = F.one_hot(labels, num_classes=2).float()
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                subgraph_val = create_balanced_subgraph(G_val, 'label')
                features_val = subgraph_val.ndata['ethereum_twitter_combined_features']
                labels_val = subgraph_val.ndata['label'].squeeze()
                logits_val = model(subgraph_val, features_val.float())
                labels_val = F.one_hot(labels_val, num_classes=2).float()
                val_loss = criterion(logits_val, labels_val)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(model)
                torch.save(model.state_dict(), 'best_model.pt')
            print(f"Epoch: {epoch + 1}/{num_epochs}, Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Testing
        best_model.eval()
        with torch.no_grad():
            subgraph_test = create_balanced_subgraph(G_test, 'label')
            features_test = subgraph_test.ndata['ethereum_twitter_combined_features']
            ground_truth = subgraph_test.ndata['label'].squeeze()

            logits_test = best_model(subgraph_test, features_test.float())
            predicted_probs = F.softmax(logits_test, dim=1)[:, 1]
            predicted_labels = (predicted_probs > 0.5).float()

            # Evaluate metrics
            auc = roc_auc_score(ground_truth.detach().numpy(), predicted_probs.detach().numpy())
            f1 = f1_score(ground_truth.detach().numpy(), predicted_labels.detach().numpy())
            precision = precision_score(ground_truth.detach().numpy(), predicted_labels.detach().numpy())
            recall = recall_score(ground_truth.detach().numpy(), predicted_labels.detach().numpy())
            accuracy = accuracy_score(ground_truth.detach().numpy(), predicted_labels.detach().numpy())
            macro_f1 = f1_score(ground_truth.detach().numpy(), predicted_labels.detach().numpy(), average='macro')
            macro_precision = precision_score(ground_truth.detach().numpy(), predicted_labels.detach().numpy(), average='macro')
            macro_recall = recall_score(ground_truth.detach().numpy(), predicted_labels.detach().numpy(), average='macro')

            print(f"AUC: {auc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}")
            print(f"Macro F1: {macro_f1:.4f}, Macro Precision: {macro_precision:.4f}, Macro Recall: {macro_recall:.4f}")


if __name__ == '__main__':
    main()
