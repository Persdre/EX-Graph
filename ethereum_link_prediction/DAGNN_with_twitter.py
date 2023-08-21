#!/usr/bin/env python
# coding: utf-8

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
import pickle as pkl
import copy

# Load graph data
with open('ethereum_with_twitter_features.pkl', 'rb') as f:
    G_dgl_with_twitter_features = pkl.load(f)

# store all edge_indices in separate files
with open('positive_train_edge_indices.pkl', 'rb') as f:
    positive_train_edge_indices = pkl.load(f)
    
with open('negative_train_edge_indices.pkl', 'rb') as f:
    negative_train_edge_indices = pkl.load(f)
    
with open('positive_validation_edge_indices.pkl', 'rb') as f:
    positive_validation_edge_indices = pkl.load(f)
    
with open('negative_validation_edge_indices.pkl', 'rb') as f:
    negative_validation_edge_indices = pkl.load(f)
    
with open('positive_test_edge_indices.pkl', 'rb') as f:
    positive_test_edge_indices = pkl.load(f)
    
with open('negative_test_edge_indices.pkl', 'rb') as f:
    negative_test_edge_indices = pkl.load(f)
    

# DAGNN Convolution Layer
class DAGNNConv(nn.Module):
    def __init__(self, in_dim, k):
        super(DAGNNConv, self).__init__()
        self.s = nn.Parameter(torch.FloatTensor(in_dim, 1))
        self.k = k
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("sigmoid")
        nn.init.xavier_uniform_(self.s, gain=gain)

    def forward(self, graph, feats):
        with graph.local_scope():
            results = [feats]
            degs = graph.in_degrees().float()
            norm = torch.pow(degs, -0.5).to(feats.device).unsqueeze(1)

            for _ in range(self.k):
                feats = feats * norm
                graph.ndata["h"] = feats
                graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
                feats = graph.ndata["h"]
                feats = feats * norm
                results.append(feats)

            H = torch.stack(results, dim=1)
            S = F.sigmoid(torch.matmul(H, self.s)).permute(0, 2, 1)
            return torch.matmul(S, H).squeeze()


# MLP Layer
class MLPLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, activation=None, dropout=0):
        super(MLPLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = 1.0 if self.activation is F.relu else nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.linear.weight, gain=gain)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, feats):
        feats = self.dropout(feats)
        feats = self.linear(feats)
        return feats if not self.activation else self.activation(feats)


# DAGNN Model
class DAGNN(nn.Module):
    def __init__(self, k, in_dim, hid_dim, out_dim, bias=True, activation=F.relu, dropout=0.1):
        super(DAGNN, self).__init__()
        self.mlp = nn.ModuleList([
            MLPLayer(in_dim, hid_dim, bias, activation, dropout),
            MLPLayer(hid_dim, hid_dim, bias, activation, dropout),
            MLPLayer(hid_dim, out_dim, bias, None, dropout)
        ])
        self.dagnn = DAGNNConv(in_dim=out_dim, k=k)

    def forward(self, graph, feats):
        for layer in self.mlp:
            feats = layer(feats)
        return self.dagnn(graph, feats)


# Generate edge embeddings
def generate_edge_embeddings(h, edges):
    src, dst = edges[0], edges[1]
    return torch.cat([h[src], h[dst]], dim=1)


# Main Training Loop
linear = nn.Linear(256, 1)
criterion = nn.BCEWithLogitsLoss()


def main():
    for i in range(5):
        model = DAGNN(1, 24, 128, 128)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        best_val_loss = float('inf')
        best_model = None
        num_epochs = 200
        patience = 20
        early_stopping_counter = 0

        for epoch in trange(num_epochs, desc="Epochs"):
            model.train()
            
            logits = model(G_dgl_with_twitter_features, G_dgl_with_twitter_features.ndata['ethereum_twitter_combined_features'].float())
            pos_train_edge_embs = generate_edge_embeddings(logits, positive_train_edge_indices)
            neg_train_edge_embs = generate_edge_embeddings(logits, negative_train_edge_indices)
            
            train_edge_embs = torch.cat([pos_train_edge_embs, neg_train_edge_embs], dim=0)
            train_edge_labels = torch.cat([torch.ones(pos_train_edge_embs.shape[0]), torch.zeros(neg_train_edge_embs.shape[0])], dim=0).unsqueeze(1)

            loss = criterion(linear(train_edge_embs), train_edge_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                logits = model(G_dgl_with_twitter_features, G_dgl_with_twitter_features.ndata['ethereum_twitter_combined_features'].float())
                pos_val_edge_embs = generate_edge_embeddings(logits, positive_validation_edge_indices)
                neg_val_edge_embs = generate_edge_embeddings(logits, negative_validation_edge_indices)
                val_edge_embs = torch.cat([pos_val_edge_embs, neg_val_edge_embs], dim=0)
                val_edge_labels = torch.cat([torch.ones(pos_val_edge_embs.shape[0]), torch.zeros(neg_val_edge_embs.shape[0])], dim=0).unsqueeze(1)
                
                val_loss = criterion(linear(val_edge_embs), val_edge_labels)
                
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
            logits = best_model(G_dgl_with_twitter_features, G_dgl_with_twitter_features.ndata['ethereum_twitter_combined_features'].float())
            pos_test_edge_embs = generate_edge_embeddings(logits, positive_test_edge_indices)
            neg_test_edge_embs = generate_edge_embeddings(logits, negative_test_edge_indices)

            test_edge_embs = torch.cat([pos_test_edge_embs, neg_test_edge_embs], dim=0)
            test_edge_labels = torch.cat([torch.ones(pos_test_edge_embs.shape[0]), torch.zeros(neg_test_edge_embs.shape[0])], dim=0).cpu().numpy()
            
            predictions = torch.sigmoid(linear(test_edge_embs)).view(-1).cpu().numpy()
            predictions_binary = (predictions > 0.5).astype(int)
            
            auc = roc_auc_score(test_edge_labels, predictions)
            f1 = f1_score(test_edge_labels, predictions_binary)
            precision = precision_score(test_edge_labels, predictions_binary)
            recall = recall_score(test_edge_labels, predictions_binary)
            accuracy = accuracy_score(test_edge_labels, predictions_binary)
            
            print(f"AUC: {auc:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()