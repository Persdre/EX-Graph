#!/usr/bin/env python
# coding: utf-8

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import APPNPConv
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
import pickle as pkl
import random
import copy

class APPNP(nn.Module):
    def __init__(self, in_feats, hiddens, n_classes, activation, feat_drop, edge_drop, alpha, k):
        super(APPNP, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(nn.Linear(in_feats, hiddens[0]))
        # hidden layers
        for i in range(1, len(hiddens)):
            self.layers.append(nn.Linear(hiddens[i - 1], hiddens[i]))
        # output layer
        self.layers.append(nn.Linear(hiddens[-1], n_classes))
        self.activation = activation
        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x
        self.propagate = APPNPConv(k, alpha, edge_drop)

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, g, features):
        h = features
        h = self.feat_drop(h)
        h = self.activation(self.layers[0](h))
        for layer in self.layers[1:-1]:
            h = self.activation(layer(h))
        h = self.layers[-1](self.feat_drop(h))
        # propagation step
        h = self.propagate(g, h)
        return h


def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = pkl.load(f)
    return data


def generate_edge_embeddings(h, edges):
    src, dst = edges[0], edges[1]
    src_embed = h[src]
    dst_embed = h[dst]
    edge_embs = torch.cat([src_embed, dst_embed], dim=1)
    return edge_embs


def main():
    # Load data
    positive_test_edge_indices = load_data('positive_test_edge_indices.pkl')
    positive_train_edge_indices = load_data('positive_train_edge_indices.pkl')
    positive_validation_edge_indices = load_data('positive_validation_edge_indices.pkl')
    negative_test_edge_indices = load_data('negative_test_edge_indices.pkl')
    negative_train_edge_indices = load_data('negative_train_edge_indices.pkl')
    negative_validation_edge_indices = load_data('negative_validation_edge_indices.pkl')
    G_dgl_training = load_data('G_dgl_training')
    
    for i in range(5):
        model = APPNP(in_feats=16, hiddens=[128, 128], n_classes=128, activation=F.relu, feat_drop=0.1, edge_drop=0, alpha=0.5, k=3)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()
        best_val_loss = float('inf')
        best_model = None
        num_epochs = 200
        patience = 10
        early_stopping_counter = 0
        
        transform = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1))

        for epoch in range(num_epochs):
            # Training
            model.train()
            logits = model(G_dgl_training, G_dgl_training.ndata['features'].float())
            pos_train_edge_embs = generate_edge_embeddings(logits, positive_train_edge_indices)
            neg_train_edge_embs = generate_edge_embeddings(logits, negative_train_edge_indices)
            train_edge_embs = torch.cat([pos_train_edge_embs, neg_train_edge_embs], dim=0)
            train_edge_labels = torch.cat([torch.ones(pos_train_edge_embs.shape[0]), torch.zeros(neg_train_edge_embs.shape[0])], dim=0).unsqueeze(1)
            loss = criterion(transform(train_edge_embs), train_edge_labels)
            print(f"Training Loss: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                logits = model(G_dgl_training, G_dgl_training.ndata['features'].float())
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
        
        # Test
        best_model.eval()
        with torch.no_grad():
            logits = best_model(G_dgl_training, G_dgl_training.ndata['features'].float())
            pos_test_edge_embs = generate_edge_embeddings(logits, positive_test_edge_indices)
            neg_test_edge_embs = generate_edge_embeddings(logits, negative_test_edge_indices)
            test_edge_embs = torch.cat([pos_test_edge_embs, neg_test_edge_embs], dim=0)
            test_edge_labels = torch.cat([torch.ones(pos_test_edge_embs.shape[0]), torch.zeros(neg_test_edge_embs.shape[0])], dim=0)
            predictions = torch.sigmoid(transform(test_edge_embs))
            predictions = predictions.view(-1).cpu().numpy()
            test_edge_labels = test_edge_labels.cpu().numpy()
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
            
            # Saving results
            with open('result.txt', 'a') as f:
                f.write(f"AUC: {auc}, F1 Score: {f1}, Precision: {precision}, Recall: {recall}, Accuracy: {accuracy}\n")

if __name__ == "__main__":
    main()
