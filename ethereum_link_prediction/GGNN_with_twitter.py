#!/usr/bin/env python
# coding: utf-8

import pickle
import dgl
import torch
import torch.nn as nn
from dgl.nn import GatedGraphConv
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
import copy


class GGSNNModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, n_steps, n_etypes):
        super(GGSNNModel, self).__init__()
        self.ggsnn_layers = nn.ModuleList([
            GatedGraphConv(in_dim, hidden_dim, n_steps, n_etypes),
            GatedGraphConv(hidden_dim, hidden_dim, n_steps, n_etypes)
        ])
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, g, features):
        h = features
        for layer in self.ggsnn_layers:
            h = layer(g, h)
        return self.fc(h)


def generate_edge_embeddings(h, edges):
    src, dst = edges[0], edges[1]
    src_embed = h[src]
    dst_embed = h[dst]
    return torch.cat([src_embed, dst_embed], dim=1)


def load_data(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def train_and_evaluate_model():
    G_dgl_with_twitter_features_converted = load_data('ethereum_with_twitter_features.pkl')

    edge_indices_files = [
        "positive_train_edge_indices.pkl",
        "negative_train_edge_indices.pkl",
        "positive_validation_edge_indices.pkl",
        "negative_validation_edge_indices.pkl",
        "positive_test_edge_indices.pkl",
        "negative_test_edge_indices.pkl"
    ]

    edge_indices = {file.split("/")[-1].split(".")[0]: load_data(file) for file in edge_indices_files}

    hidden_dim, num_classes, n_steps, n_etypes = 128, 128, 5, 1

    transform = nn.Sequential(
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    )

    for i in range(5):
        model = GGSNNModel(24, hidden_dim, num_classes, n_steps, n_etypes)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()
        best_model, best_val_loss = None, float('inf')
        early_stopping_counter, num_epochs, patience = 0, 200, 20

        for epoch in range(num_epochs):
            model.train()
            logits = model(G_dgl_with_twitter_features_converted, G_dgl_with_twitter_features_converted.ndata['ethereum_twitter_combined_features'].float())
            
            pos_train_edge_embs = generate_edge_embeddings(logits, edge_indices["positive_train_edge_indices"])
            neg_train_edge_embs = generate_edge_embeddings(logits, edge_indices["negative_train_edge_indices"])

            train_edge_embs = torch.cat([pos_train_edge_embs, neg_train_edge_embs], dim=0)
            train_edge_labels = torch.cat([torch.ones(pos_train_edge_embs.shape[0]), torch.zeros(neg_train_edge_embs.shape[0])], dim=0).unsqueeze(1)

            loss = criterion(transform(train_edge_embs), train_edge_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                logits = model(G_dgl_with_twitter_features_converted, G_dgl_with_twitter_features_converted.ndata['ethereum_twitter_combined_features'].float())
                
                pos_val_edge_embs = generate_edge_embeddings(logits, edge_indices["positive_validation_edge_indices"])
                neg_val_edge_embs = generate_edge_embeddings(logits, edge_indices["negative_validation_edge_indices"])

                val_edge_embs = torch.cat([pos_val_edge_embs, neg_val_edge_embs], dim=0)
                val_edge_labels = torch.cat([torch.ones(pos_val_edge_embs.shape[0]), torch.zeros(neg_val_edge_embs.shape[0])], dim=0).unsqueeze(1)

                val_loss = criterion(transform(val_edge_embs), val_edge_labels)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stopping_counter = 0
                    best_model = copy.deepcopy(model)
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= patience:
                        break

        best_model.eval()
        with torch.no_grad():
            logits = best_model(G_dgl_with_twitter_features_converted, G_dgl_with_twitter_features_converted.ndata['ethereum_twitter_combined_features'].float())
            
            pos_test_edge_embs = generate_edge_embeddings(logits, edge_indices["positive_test_edge_indices"])
            neg_test_edge_embs = generate_edge_embeddings(logits, edge_indices["negative_test_edge_indices"])

            test_edge_embs = torch.cat([pos_test_edge_embs, neg_test_edge_embs], dim=0)
            test_edge_labels = torch.cat([torch.ones(pos_test_edge_embs.shape[0]), torch.zeros(neg_test_edge_embs.shape[0])], dim=0)

            predictions = torch.sigmoid(transform(test_edge_embs))
            predictions_binary = (predictions > 0.5).astype(int)
            metrics = {
                "AUC": roc_auc_score(test_edge_labels, predictions),
                "F1 Score": f1_score(test_edge_labels, predictions_binary),
                "Precision": precision_score(test_edge_labels, predictions_binary),
                "Recall": recall_score(test_edge_labels, predictions_binary),
                "Accuracy": accuracy_score(test_edge_labels, predictions_binary)
            }

            with open('results_with_twitter.txt', 'a') as f:
                for metric, value in metrics.items():
                    f.write(f"{metric}: {value}\n")
                f.write('\n')


if __name__ == '__main__':
    train_and_evaluate_model()
