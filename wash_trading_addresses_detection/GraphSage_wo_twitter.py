#!/usr/bin/env python
# coding: utf-8

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning
import warnings
import copy
import numpy as np
import random
import pickle as pkl

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


class GraphSageModel(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, dropout_rate=0.1):
        super(GraphSageModel, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, aggregator_type='mean')
        self.conv2 = SAGEConv(h_feats, h_feats, aggregator_type='mean')
        self.conv3 = SAGEConv(h_feats, num_classes, aggregator_type='mean')
        self.dropout = nn.Dropout(dropout_rate)
        self.batchnorm = nn.BatchNorm1d(h_feats)

    def forward(self, graph, x):
        h = self.conv1(graph, x)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.batchnorm(h)
        h = self.conv2(graph, h)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv3(graph, h)
        return h


def train_and_evaluate(data, model, optimizer, criterion, num_epochs=100, patience=20, filename='results.txt'):
    best_val_loss = float('inf')
    best_model = None
    
    G_train_dgl, G_val_dgl, G_test_dgl = data

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(G_train_dgl, G_train_dgl.ndata['ethereum_ethereum_features'])
        labels = F.one_hot(G_train_dgl.ndata['label'], num_classes=2).float()
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(G_val_dgl, G_val_dgl.ndata['ethereum_features'])
            labels = F.one_hot(G_val_dgl.ndata['label'], num_classes=2).float()
            val_loss = criterion(logits, labels)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(model)
                torch.save(model.state_dict(), 'best_model.pt')

            print(f"Epoch: {epoch + 1}/{num_epochs}, Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}")

    best_model.eval()
    with torch.no_grad():
        logits = best_model(G_test_dgl, G_test_dgl.ndata['ethereum_features'])
        _, predicted = torch.max(logits, 1)
        true = G_test_dgl.ndata['label']
        
        # Compute metrics
        auc = roc_auc_score(true, logits[:, 1])
        f1 = f1_score(true, predicted)
        precision = precision_score(true, predicted)
        recall = recall_score(true, predicted)
        accuracy = accuracy_score(true, predicted)
        
        # Store results
        with open(filename, "a") as f:
            f.write(f"AUC: {auc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}\n")

        print(f"AUC: {auc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}")


def main():
    # Load datasets
    with open('G_train_dgl.gpickle', 'rb') as f:
        G_train_dgl = pkl.load(f)
    with open('G_val_dgl.gpickle', 'rb') as f:
        G_val_dgl = pkl.load(f)
    with open('G_test_dgl.gpickle', 'rb') as f:
        G_test_dgl = pkl.load(f)

    # Parameters
    lr = 0.001
    num_epochs = 200
    patience = 20
    
    model = GraphSageModel(in_feats=8, h_feats=128, num_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    train_and_evaluate((G_train_dgl, G_val_dgl, G_test_dgl), model, optimizer, criterion, num_epochs, patience)


if __name__ == "__main__":
    main()
