#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import dgl
import pickle as pkl
from dgl.nn.pytorch.conv import TAGConv
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
import copy

class TAGCN(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super(TAGCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.layers.append(TAGConv(in_feats, n_hidden, activation=activation))
        for i in range(n_layers - 1):
            self.layers.append(TAGConv(n_hidden, n_hidden, activation=activation))
        self.layers.append(TAGConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
        return h

def load_data(filename):
    with open(filename, 'rb') as f:
        return pkl.load(f)

def generate_edge_embeddings(h, edges):
    src, dst = edges[0], edges[1]
    src_embed = h[src]
    dst_embed = h[dst]
    return torch.cat([src_embed, dst_embed], dim=1)

def main():
    matching_link_prediction_graph.pkl = load_data('matching_link_prediction_graph.pkl')

    positive_test_edge_indices = load_data('positive_test_edge_indices.pkl')
    positive_train_edge_indices = load_data('positive_train_edge_indices.pkl')
    positive_validation_edge_indices = load_data('positive_validation_edge_indices.pkl')
    negative_test_edge_indices = load_data('negative_test_edge_indices.pkl')
    negative_train_edge_indices = load_data('negative_train_edge_indices.pkl')
    negative_validation_edge_indices = load_data('negative_validation_edge_indices.pkl')

    transform = nn.Sequential(
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    )

    for _ in range(5):
        model = TAGCN(matching_link_prediction_graph.pkl, in_feats=16, n_hidden=128, n_classes=128, n_layers=1, activation=torch.nn.functional.relu, dropout=0.1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()
        best_val_loss = float('inf')
        best_model = None
        num_epochs = 200
        patience = 100
        early_stopping_counter = 0

        for epoch in range(num_epochs):
            # Training phase
            model.train()
            logits = model(matching_link_prediction_graph.pkl.ndata['combined_features'].float())
            pos_train_edge_embs = generate_edge_embeddings(logits, positive_train_edge_indices)
            neg_train_edge_embs = generate_edge_embeddings(logits, negative_train_edge_indices)
            train_edge_embs = torch.cat([pos_train_edge_embs, neg_train_edge_embs], dim=0)
            train_edge_labels = torch.cat([torch.ones(pos_train_edge_embs.shape[0]), torch.zeros(neg_train_edge_embs.shape[0])], dim=0).unsqueeze(1)
            loss = criterion(transform(train_edge_embs), train_edge_labels)
            print(f"Training Loss: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Validation phase
            model.eval()
            with torch.no_grad():
                logits = model(matching_link_prediction_graph.pkl.ndata['combined_features'].float())
                pos_val_edge_embs = generate_edge_embeddings(logits, positive_validation_edge_indices)
                neg_val_edge_embs = generate_edge_embeddings(logits, negative_validation_edge_indices)
                val_edge_embs = torch.cat([pos_val_edge_embs, neg_val_edge_embs], dim=0)
                val_edge_labels = torch.cat([torch.ones(pos_val_edge_embs.shape[0]), torch.zeros(neg_val_edge_embs.shape[0])], dim=0).unsqueeze(1)
                val_loss = criterion(transform(val_edge_embs), val_edge_labels)
                print(f"Validation Loss: {val_loss.item()}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stopping_counter = 0
                    best_model = copy.deepcopy(model)
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= patience:
                        print("Early Stopping!")
                        break

        # Testing phase
        best_model.eval()
        with torch.no_grad():
            logits = best_model(matching_link_prediction_graph.pkl.ndata['combined_features'].float())
            pos_test_edge_embs = generate_edge_embeddings(logits, positive_test_edge_indices)
            neg_test_edge_embs = generate_edge_embeddings(logits, negative_test_edge_indices)
            test_edge_embs = torch.cat([pos_test_edge_embs, neg_test_edge_embs], dim=0)
            test_edge_labels = torch.cat([torch.ones(pos_test_edge_embs.shape[0]), torch.zeros(neg_test_edge_embs.shape[0])], dim=0)
            predictions = torch.sigmoid(transform(test_edge_embs))
            predictions = predictions.view(-1).cpu().numpy()
            test_edge_labels = test_edge_labels.cpu().numpy()
            # Compute metrics
            auc = roc_auc_score(test_edge_labels, predictions)
            predictions_binary = (predictions > 0.5).astype(int)
            f1 = f1_score(test_edge_labels, predictions_binary)
            precision = precision_score(test_edge_labels, predictions_binary)
            recall = recall_score(test_edge_labels, predictions_binary)
            accuracy = accuracy_score(test_edge_labels, predictions_binary)
            print(f"AUC: {auc}")
            print(f"F1 Score: {f1}")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"Accuracy: {accuracy}")

            with open('result.txt', 'a') as f:
                f.write(f"AUC: {auc}, F1 Score: {f1}, Precision: {precision}, Recall: {recall}, Accuracy: {accuracy}\n")

if __name__ == '__main__':
    main()
