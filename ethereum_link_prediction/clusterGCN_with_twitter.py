#!/usr/bin/env python
# coding: utf-8

import time
import dgl
import dgl.nn as dglnn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.nodeproppred import DglNodePropPredDataset
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
import copy
import pickle as pkl

with open('ethereum_with_twitter_features.pkl', 'rb') as f:
    G_dgl_with_twitter_features_converted = pkl.load(f)
    
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


# print some nodes and features of it
print(G_dgl_with_twitter_features_converted.nodes[0])

# print another example
print(G_dgl_with_twitter_features_converted.nodes[1].data['ethereum_twitter_combined_features'])

class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, "mean"))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, "mean"))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, "mean"))
        self.dropout = nn.Dropout(0.1)

    def forward(self, sg, x):
        h = x
        for l, layer in enumerate(self.layers):
            h = layer(sg, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

def generate_edge_embeddings(h, edges):
    # Extract the source and target node indices from the edges
    src, dst = edges[0], edges[1]
    
    # Use the node indices to get the corresponding node embeddings
    src_embed = h[src]
    dst_embed = h[dst]

    # Concatenate the source and target node embeddings
    edge_embs = torch.cat([src_embed, dst_embed], dim=1)

    return edge_embs


# write a for loop to get 5 results
# store the results in a list
def main():
    for i in range(5):
        model = SAGE(16, 128, 128)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()
        best_val_loss = float('inf')
        best_model = None
        
        num_epochs = 200
        patience = 30
        early_stopping_counter = 0
        
        transform = nn.Sequential(
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 1))
        
        for epoch in range(num_epochs):
            model.train()
            logits = model(G_dgl_with_twitter_features_converted, G_dgl_with_twitter_features_converted.ndata['ethereum_twitter_combined_features'].float())
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
                logits = model(G_dgl_with_twitter_features_converted, G_dgl_with_twitter_features_converted.ndata['ethereum_twitter_combined_features'].float())
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
            logits = best_model(G_dgl_with_twitter_features_converted, G_dgl_with_twitter_features_converted.ndata['ethereum_twitter_combined_features'].float())
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
        with open('result_with_twitter.txt', 'a') as f:
            # write auc, f1, precision, recall
            f.write(f"AUC: {auc}, F1 Score: {f1}, Precision: {precision}, Recall: {recall}, Accuracy: {accuracy}\n")        

if __name__ == '__main__':
    main()