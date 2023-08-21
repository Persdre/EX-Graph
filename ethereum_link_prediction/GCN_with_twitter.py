#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
import dgl
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
import pickle
import copy

# Define the GCN model
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
        x = self.dropout(x)  
        x = self.batchnorm1(x)
        x = F.relu(self.conv2(g, x))
        x = self.dropout(x)
        x = self.conv3(g, x)
        return x

def generate_edge_embeddings(h, edges):
    src, dst = edges[0], edges[1]
    src_embed = h[src]
    dst_embed = h[dst]
    edge_embs = torch.cat([src_embed, dst_embed], dim=1)
    return edge_embs

# Load graph data
with open('ethereum_with_twitter_features.pkl', 'rb') as f:
    G_dgl_with_twitter_features_converted = pickle.load(f)

# Load edge indices
data_paths = [
    'positive_train_edge_indices.pkl', 
    'negative_train_edge_indices.pkl', 
    'positive_validation_edge_indices.pkl',
    'negative_validation_edge_indices.pkl',
    'positive_test_edge_indices.pkl',
    'negative_test_edge_indices.pkl'
]
edge_indices_data = {}

for path in data_paths:
    with open(f'{path}', 'rb') as f:
        key = path.split('/')[-1].split('.')[0]
        edge_indices_data[key] = pickle.load(f)

linear = nn.Linear(256, 1)


def main():
    # Training loop
    for i in range(5):
        model = GCN(24, 128, 128, 0.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
        criterion = nn.BCEWithLogitsLoss()
        best_val_loss = float('inf')
        best_model = None
        num_epochs = 200
        patience = 20
        early_stopping_counter = 0
        
        for epoch in range(num_epochs):
            # Training
            model.train()
            logits = model(G_dgl_with_twitter_features_converted, G_dgl_with_twitter_features_converted.ndata['ethereum_twitter_combined_features'].float())
            pos_train_edge_embs = generate_edge_embeddings(logits, edge_indices_data['positive_train_edge_indices'])
            neg_train_edge_embs = generate_edge_embeddings(logits, edge_indices_data['negative_train_edge_indices'])
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
                logits = model(G_dgl_with_twitter_features_converted, G_dgl_with_twitter_features_converted.ndata['ethereum_twitter_combined_features'].float())
                pos_val_edge_embs = generate_edge_embeddings(logits, edge_indices_data['positive_validation_edge_indices'])
                neg_val_edge_embs = generate_edge_embeddings(logits, edge_indices_data['negative_validation_edge_indices'])
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

        # Test
        best_model.eval()
        with torch.no_grad():
            logits = best_model(G_dgl_with_twitter_features_converted, G_dgl_with_twitter_features_converted.ndata['ethereum_twitter_combined_features'].float())
            pos_test_edge_embs = generate_edge_embeddings(logits, edge_indices_data['positive_test_edge_indices'])
            neg_test_edge_embs = generate_edge_embeddings(logits, edge_indices_data['negative_test_edge_indices'])
            test_edge_embs = torch.cat([pos_test_edge_embs, neg_test_edge_embs], dim=0)
            test_edge_labels = torch.cat([torch.ones(pos_test_edge_embs.shape[0]), torch.zeros(neg_test_edge_embs.shape[0])], dim=0).cpu().numpy()

            predictions = torch.sigmoid(linear(test_edge_embs)).view(-1).cpu().numpy()
            predictions_binary = (predictions > 0.5).astype(int)
            auc = roc_auc_score(test_edge_labels, predictions)
            f1 = f1_score(test_edge_labels, predictions_binary)
            precision = precision_score(test_edge_labels, predictions_binary)
            recall = recall_score(test_edge_labels, predictions_binary)
            accuracy = accuracy_score(test_edge_labels, predictions_binary)

            # Output results
            print(f"AUC: {auc}\nF1 Score: {f1}\nPrecision: {precision}\nRecall: {recall}\nAccuracy: {accuracy}")
            with open('results_with_twitter.txt', 'a') as f:
                f.write(f"AUC: {auc}\nF1 Score: {f1}\nPrecision: {precision}\nRecall: {recall}\nAccuracy: {accuracy}\n\n")

if __name__ == '__main__':
    main()