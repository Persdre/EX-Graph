#!/usr/bin/env python
# coding: utf-8

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
import pickle as pkl
from copy import deepcopy
from tqdm import trange

# Load the training data
with open('G_dgl_training', 'rb') as f:
    G_dgl_training = pkl.load(f)

print('number of nodes in G_dgl_training: ', G_dgl_training.number_of_nodes())


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
            norm = torch.pow(degs, -0.5)
            norm = norm.to(feats.device).unsqueeze(1)

            for _ in range(self.k):
                feats = feats * norm
                graph.ndata["h"] = feats
                graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
                feats = graph.ndata["h"]
                feats = feats * norm
                results.append(feats)

            H = torch.stack(results, dim=1)
            S = torch.sigmoid(torch.matmul(H, self.s))
            S = S.permute(0, 2, 1)
            H = torch.matmul(S, H).squeeze()
            return H


class MLPLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, activation=None, dropout=0):
        super(MLPLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = 1.0
        if self.activation is F.relu:
            gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.linear.weight, gain=gain)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, feats):
        feats = self.dropout(feats)
        feats = self.linear(feats)
        if self.activation:
            feats = self.activation(feats)
        return feats


class DAGNN(nn.Module):
    def __init__(self, k, in_dim, hid_dim, out_dim, bias=True, activation=F.relu, dropout=0.1):
        super(DAGNN, self).__init__()
        self.mlp = nn.ModuleList([
            MLPLayer(in_dim=in_dim, out_dim=hid_dim, bias=bias, activation=activation, dropout=dropout),
            MLPLayer(in_dim=hid_dim, out_dim=hid_dim, bias=bias, activation=activation, dropout=dropout),
            MLPLayer(in_dim=hid_dim, out_dim=out_dim, bias=bias, activation=None, dropout=dropout)
        ])
        self.dagnn = DAGNNConv(in_dim=out_dim, k=k)

    def forward(self, graph, feats):
        for layer in self.mlp:
            feats = layer(feats)
        feats = self.dagnn(graph, feats)
        return feats


def generate_edge_embeddings(h, edges):
    src, dst = edges[0], edges[1]
    src_embed = h[src]
    dst_embed = h[dst]
    edge_embs = torch.cat([src_embed, dst_embed], dim=1)
    return edge_embs


def main():
    with open('positive_test_edge_indices.pkl', 'rb') as f:
        positive_test_edge_indices = pkl.load(f)

    with open('positive_train_edge_indices.pkl', 'rb') as f:
        positive_train_edge_indices = pkl.load(f)

    with open('positive_validation_edge_indices.pkl', 'rb') as f:
        positive_validation_edge_indices = pkl.load(f)

    with open('negative_test_edge_indices.pkl', 'rb') as f:
        negative_test_edge_indices = pkl.load(f)

    with open('negative_train_edge_indices.pkl', 'rb') as f:
        negative_train_edge_indices = pkl.load(f)

    with open('negative_validation_edge_indices.pkl', 'rb') as f:
        negative_validation_edge_indices = pkl.load(f)

    # Define a model
    model = DAGNN(1, 16, 128, 128)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    best_val_loss = float('inf')
    best_model = None
    num_epochs = 200
    patience = 20
    early_stopping_counter = 0

    transform = nn.Sequential(
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 1))

    # Main training loop
    for i in range(5):
        model = DAGNN(1,16,128,128)
        # model = model.to('cuda:1')
        # train on positive edges, negative edges; also use validation edges to stop epochs
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        criterion = nn.BCEWithLogitsLoss()

        best_val_loss = float('inf')
        best_model = None
        num_epochs = 200
        patience = 100
        early_stopping_counter = 0
        
        
        transform = nn.Sequential(
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 1))
        
        for epoch in range(num_epochs):
            model.train()
            
            # forward pass
            logits = model(G_dgl_training, G_dgl_training.ndata['features'].float())
            
            # generate edge embeddings
            pos_train_edge_embs = generate_edge_embeddings(logits, positive_train_edge_indices)
            neg_train_edge_embs = generate_edge_embeddings(logits, negative_train_edge_indices)
            
            # concatenete positive and negative edge embeddings
            train_edge_embs = torch.cat([pos_train_edge_embs, neg_train_edge_embs], dim=0)
            train_edge_labels = torch.cat([torch.ones(pos_train_edge_embs.shape[0]), torch.zeros(neg_train_edge_embs.shape[0])], dim=0).unsqueeze(1)
            
            # print shapes of tensors for debugging
            # print(f"Train Edge Embeddings Shape: {train_edge_embs.shape}")
            # print(f"Train Edge Labels Shape: {train_edge_labels.shape}")
            
            # calculate loss
            loss = criterion(transform(train_edge_embs), train_edge_labels)
            print(f"Training Loss: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
            # validation
            model.eval()
            
            with torch.no_grad():
                # repeat the same process as above for validation samples
                logits = model(G_dgl_training, G_dgl_training.ndata['features'].float())
                pos_val_edge_embs = generate_edge_embeddings(logits, positive_validation_edge_indices)
                neg_val_edge_embs = generate_edge_embeddings(logits, negative_validation_edge_indices)
                val_edge_embs = torch.cat([pos_val_edge_embs, neg_val_edge_embs], dim=0)
                val_edge_labels = torch.cat([torch.ones(pos_val_edge_embs.shape[0]), torch.zeros(neg_val_edge_embs.shape[0])], dim=0).unsqueeze(1)
                # # print shapes of tensors for debugging
                # print(f"Validation Edge Embeddings Shape: {val_edge_embs.shape}")
                # print(f"Validation Edge Labels Shape: {val_edge_labels.shape}")

                val_loss = criterion(transform(val_edge_embs), val_edge_labels)
                print(f"Validation Loss: {val_loss.item()}")
                
                # early stopping based on validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    # add patience
                    early_stopping_counter = 0
                    # # save the best model
                    best_model = copy.deepcopy(model)
                
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= patience:
                        print("Early Stopping!")
                        break
                    
        # switch to evaluation mode
        best_model.eval()

        with torch.no_grad():
            # generate the embeddings using the best model
            logits = best_model(G_dgl_training, G_dgl_training.ndata['features'].float())

            # generate edge embeddings for the test samples
            pos_test_edge_embs = generate_edge_embeddings(logits, positive_test_edge_indices)
            neg_test_edge_embs = generate_edge_embeddings(logits, negative_test_edge_indices)

            # concatenate the positive and negative edge embeddings and labels
            test_edge_embs = torch.cat([pos_test_edge_embs, neg_test_edge_embs], dim=0)
            test_edge_labels = torch.cat([torch.ones(pos_test_edge_embs.shape[0]), torch.zeros(neg_test_edge_embs.shape[0])], dim=0)
            
            predictions = torch.sigmoid(transform(test_edge_embs))
            
            # reshape the predictions and the labels
            predictions = predictions.view(-1).cpu().numpy()
            test_edge_labels = test_edge_labels.cpu().numpy()
            
            auc = roc_auc_score(test_edge_labels, predictions)
            # here use 0.5 as threshold
            predictions_binary = (predictions > 0.5).astype(int)
            f1 = f1_score(test_edge_labels, predictions_binary)
            precision = precision_score(test_edge_labels, predictions_binary)
            recall = recall_score(test_edge_labels, predictions_binary)
            accuracy = accuracy_score(test_edge_labels, predictions_binary)
        # also record loss
        # print(f"Test Loss: {criterion(transform(test_edge_embs), test_edge_labels)}")
        print(f"AUC: {auc}")
        print(f"F1 Score: {f1}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"Accuracy: {accuracy}")
        

    # Write results to file
    with open('result.txt', 'a') as f:
        f.write(f"AUC: {auc}, F1 Score: {f1}, Precision: {precision}, Recall: {recall}, Accuracy: {accuracy}\n")

if __name__ == "__main__":
    main()
