#!/usr/bin/env python
# coding: utf-8

import pickle as pkl
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import APPNPConv, GATConv, GraphConv
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
import numpy as np
import torch.backends.cudnn as cudnn
import random
import copy

class GGSNNModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, n_steps, n_etypes):
        super(GGSNNModel, self).__init__()
        self.ggsnn_layers = nn.ModuleList()
        self.ggsnn_layers.append(GatedGraphConv(in_dim, hidden_dim, n_steps, n_etypes))
        self.ggsnn_layers.append(GatedGraphConv(hidden_dim, hidden_dim, n_steps, n_etypes))
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, g, features):
        h = features
        for layer in self.ggsnn_layers:
            h = layer(g, h)
        h = self.fc(h)
        return h


# read all edge_indices in separate files
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


# read ethereum_with_twitter_features.pkl
with open('ethereum_with_twitter_features.pkl', 'rb') as f:
    G_dgl = pkl.load(f)


def generate_edge_embeddings(h, edges):
    # Extract the source and target node indices from the edges
    src, dst = edges[0], edges[1]
    
    # Use the node indices to get the corresponding node embeddings
    src_embed = h[src]
    dst_embed = h[dst]

    # Concatenate the source and target node embeddings
    edge_embs = torch.cat([src_embed, dst_embed], dim=1)

    return edge_embs



hidden_dim = 128
num_classes = 128
n_steps = 5
n_etypes = 1



# write a for loop to get 5 results
# store the results in a txt file
def main():
    for i in range(5):
        model = GGSNNModel(8, hidden_dim, num_classes, n_steps, n_etypes)
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
            model.train()
            logits = model(G_dgl, G_dgl.ndata['ethereum_features'])
            pos_train_edge_embs = generate_edge_embeddings(logits, positive_train_edge_indices)
            neg_train_edge_embs = generate_edge_embeddings(logits, negative_train_edge_indices)
            train_edge_embs = torch.cat([pos_train_edge_embs, neg_train_edge_embs], dim=0)
            train_edge_labels = torch.cat([torch.ones(pos_train_edge_embs.shape[0]), torch.zeros(neg_train_edge_embs.shape[0])], dim=0).unsqueeze(1)
            
            
            loss = criterion(transform(train_edge_embs), train_edge_labels)
            print(f"Training Loss: {loss.item()}")
            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()
            
            # validation
            model.eval()
            with torch.no_grad():
                logits = model(G_dgl, G_dgl.ndata['ethereum_features'].float())
                pos_val_edge_embs = generate_edge_embeddings(logits, positive_validation_edge_indices)
                neg_val_edge_embs = generate_edge_embeddings(logits, negative_validation_edge_indices)
                val_edge_embs = torch.cat([pos_val_edge_embs, neg_val_edge_embs], dim=0)
                val_edge_labels = torch.cat([torch.ones(pos_val_edge_embs.shape[0]), torch.zeros(neg_val_edge_embs.shape[0])], dim=0).unsqueeze(1)
                
                val_loss = criterion(transform(val_edge_embs), val_edge_labels)
                print (f"Validation Loss: {val_loss.item()}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = copy.deepcopy(model)
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= patience:
                        print('early stopping due to validation loss not improving')
                        break
                    
        # switch to evaluation mode
        best_model.eval()
        
        with torch.no_grad():
            # generate the embeddings using the best model
            logits = best_model(G_dgl, G_dgl.ndata['ethereum_features'].float())
            positive_test_edge_embs = generate_edge_embeddings(logits, positive_test_edge_indices)
            negative_test_edge_embs = generate_edge_embeddings(logits, negative_test_edge_indices)
            test_edge_embs = torch.cat([positive_test_edge_embs, negative_test_edge_embs], dim=0)
            test_edge_labels = torch.cat([torch.ones(positive_test_edge_embs.shape[0]), torch.zeros(negative_test_edge_embs.shape[0])], dim=0)
            
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
        
        # write the result to a txt file
        with open('result_wo_twitter.txt', 'a') as f:
            # write auc, f1, precision, recall
            f.write(f"AUC: {auc}, F1 Score: {f1}, Precision: {precision}, Recall: {recall}, Accuracy: {accuracy}\n")        
            
            
if __name__ == '__main__':
    main()

