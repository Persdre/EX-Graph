#!/usr/bin/env python
# coding: utf-8

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning
import warnings
import pickle
import copy
import numpy as np


# Load the DGL graph
with open('ethereum_with_twitter_features.pkl', 'rb') as f:
    G_dgl = pickle.load(f)


class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, dropout_rate):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, hidden_size)
        self.conv3 = GraphConv(hidden_size, out_feats)
        self.dropout = nn.Dropout(dropout_rate)
        self.batchnorm1 = nn.BatchNorm1d(hidden_size)

    def forward(self, g, features):
        x = F.relu(self.conv1(g, features))
        x = self.dropout(self.batchnorm1(x))
        x = F.relu(self.conv2(g, x))
        x = self.dropout(x)
        x = self.conv3(g, x)
        return x


# Load edge indices
def load_edges(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


positive_train_edge_indices = load_edges('positive_train_edge_indices.pkl')
negative_train_edge_indices = load_edges('negative_train_edge_indices.pkl')
positive_validation_edge_indices = load_edges('positive_validation_edge_indices.pkl')
negative_validation_edge_indices = load_edges('negative_validation_edge_indices.pkl')
positive_test_edge_indices = load_edges('positive_test_edge_indices.pkl')
negative_test_edge_indices = load_edges('negative_test_edge_indices.pkl')


def generate_edge_embeddings(h, edges):
    src, dst = edges
    return torch.cat([h[src], h[dst]], dim=1)

linear = nn.Linear(256, 1)

def main():
    for i in range(5):
        model = GCN(8, 128, 128, 0.1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()
        best_val_loss = float('inf')
        best_model = None
        num_epochs = 200
        patience = 20
        early_stopping_counter = 0

        for epoch in range(num_epochs):
            model.train()
            logits = model(G_dgl, G_dgl.ndata['ethereum_features'])

            pos_train_edge_embs = generate_edge_embeddings(logits, positive_train_edge_indices)
            neg_train_edge_embs = generate_edge_embeddings(logits, negative_train_edge_indices)
            train_edge_embs = torch.cat([pos_train_edge_embs, neg_train_edge_embs], dim=0)
            train_edge_labels = torch.cat([torch.ones(pos_train_edge_embs.shape[0]), torch.zeros(neg_train_edge_embs.shape[0])], dim=0).unsqueeze(1)

            loss = criterion(linear(train_edge_embs), train_edge_labels)
            print(f"Training Loss: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                logits = model(G_dgl, G_dgl.ndata['ethereum_features'])
                pos_val_edge_embs = generate_edge_embeddings(logits, positive_validation_edge_indices)
                neg_val_edge_embs = generate_edge_embeddings(logits, negative_validation_edge_indices)
                val_edge_embs = torch.cat([pos_val_edge_embs, neg_val_edge_embs], dim=0)
                val_edge_labels = torch.cat([torch.ones(pos_val_edge_embs.shape[0]), torch.zeros(neg_val_edge_embs.shape[0])], dim=0).unsqueeze(1)

                val_loss = criterion(linear(val_edge_embs), val_edge_labels)
                print(f"Validation Loss: {val_loss.item()}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stopping_counter = 0
                    best_model = copy.deepcopy(model)
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= patience:
                        print('Early stopping due to validation loss not improving')
                        break

        best_model.eval()
        with torch.no_grad():
            logits = best_model(G_dgl, G_dgl.ndata['ethereum_features'])
            pos_test_edge_embs = generate_edge_embeddings(logits, positive_test_edge_indices)
            neg_test_edge_embs = generate_edge_embeddings(logits, negative_test_edge_indices)
            test_edge_embs = torch.cat([pos_test_edge_embs, neg_test_edge_embs], dim=0)
            test_edge_labels = torch.cat([torch.ones(pos_test_edge_embs.shape[0]), torch.zeros(neg_test_edge_embs.shape[0])], dim=0)

            predictions = torch.sigmoid(linear(test_edge_embs))
            predictions_binary = (predictions > 0.5).astype(int)
            test_edge_labels = test_edge_labels.cpu().numpy()
            metrics = {
                'AUC': roc_auc_score(test_edge_labels, predictions),
                'F1 Score': f1_score(test_edge_labels, predictions_binary),
                'Precision': precision_score(test_edge_labels, predictions_binary),
                'Recall': recall_score(test_edge_labels, predictions_binary),
                'Accuracy': accuracy_score(test_edge_labels, predictions_binary)
            }

            with open('results_wo_twitter.txt', 'a') as f:
                for metric, value in metrics.items():
                    print(f"{metric}: {value}")
                    f.write(f"{metric}: {value}\n")
                f.write('\n')
                
if __name__ == '__main__':
    main()

