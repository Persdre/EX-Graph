#!/usr/bin/env python
# coding: utf-8


import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np
import torch.backends.cudnn as cudnn
import random
import pickle as pkl
import copy

# load edge indices for train, val, test
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


# load G_dgl for training
with open('ethereum_with_twitter_features.pkl', 'rb') as f:
    G_dgl = pkl.load(f)


class GATModel(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, dropout=0.1):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, num_heads=num_heads)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, num_heads=num_heads)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.conv3 = GATConv(hidden_dim * num_heads, out_dim, num_heads=1)

    def forward(self, g, h):
        # First Layer
        h = self.conv1(g, h).flatten(1)
        h = F.elu(self.dropout1(h))
        # Second Layer
        h = self.conv2(g, h).flatten(1)
        h = F.elu(self.dropout2(h))
        # Third Layer
        h = self.conv3(g, h).mean(1)
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

transform = nn.Sequential(
    nn.Linear(256, 128), # 256 input features, 128 output features
    nn.ReLU(),
    nn.Linear(128, 1), # 128 input features, 16 output features
)

def main():
    for i in range(5):
        model = GATModel(8, 128, 128, 3)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
        criterion = nn.BCEWithLogitsLoss()
        best_val_loss = float('inf')
        best_model = None
        num_epochs = 200
        patience = 10
        early_stopping_counter = 0
        
        for epoch in range(num_epochs):
            model.train()
            
            # forward pass
            logits = model(G_dgl, G_dgl.ndata['ethereum_features'].float())
            
            # generate edge embeddings
            pos_train_edge_embs = generate_edge_embeddings(logits, positive_train_edge_indices)
            neg_train_edge_embs = generate_edge_embeddings(logits, negative_train_edge_indices)
            
            # concatenete positive and negative edge embeddings
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
                # repeat the same process as above for validation samples
                logits = model(G_dgl, G_dgl.ndata['ethereum_features'].float())
                pos_val_edge_embs = generate_edge_embeddings(logits, positive_validation_edge_indices)
                neg_val_edge_embs = generate_edge_embeddings(logits, negative_validation_edge_indices)
                val_edge_embs = torch.cat([pos_val_edge_embs, neg_val_edge_embs], dim=0)
                val_edge_labels = torch.cat([torch.ones(pos_val_edge_embs.shape[0]), torch.zeros(neg_val_edge_embs.shape[0])], dim=0).unsqueeze(1)
                # print shapes of tensors for debugging
                # print(f"Validation Edge Embeddings Shape: {val_edge_embs.shape}")
                # print(f"Validation Edge Labels Shape: {val_edge_labels.shape}")

                val_loss = criterion(transform(val_edge_embs), val_edge_labels)
                print(f"Validation Loss: {val_loss.item()}")
                
                # early stopping based on validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stopping_counter = 0
                    # save the best model
                    best_model = copy.deepcopy(model)
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

            # generate edge embeddings for the test samples
            pos_test_edge_embs = generate_edge_embeddings(logits, positive_test_edge_indices)
            neg_test_edge_embs = generate_edge_embeddings(logits, negative_test_edge_indices)

            # concatenate the positive and negative edge embeddings and labels
            test_edge_embs = torch.cat([pos_test_edge_embs, neg_test_edge_embs], dim=0)
            test_edge_labels = torch.cat([torch.ones(pos_test_edge_embs.shape[0]), torch.zeros(neg_test_edge_embs.shape[0])], dim=0)


            # test_loss = criterion(linear(test_edge_embs), val_edge_labels)
            # calculate predictions using the linear layer
            
            predictions = torch.sigmoid(transform(test_edge_embs))
            
            # reshape the predictions and the labels
            predictions = predictions.view(-1).cpu().numpy()
            test_edge_labels = test_edge_labels.cpu().numpy()

            # document the results, including auc, f1, precision, recall, accuracy, average_precision
            auc = roc_auc_score(test_edge_labels, predictions)
            predictions_binary = (predictions > 0.5).astype(int)
            f1 = f1_score(test_edge_labels, predictions_binary)
            precision = precision_score(test_edge_labels, predictions_binary)
            recall = recall_score(test_edge_labels, predictions_binary)
            accuracy = accuracy_score(test_edge_labels, predictions_binary)
            average_precision = average_precision_score(test_edge_labels, predictions_binary)
            

            print(f"AUC: {auc}")
            print(f"F1 Score: {f1}")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"Accuracy: {accuracy}")
            print(f"Average Precision Score: {average_precision}")
        # print accuracy, f1, precision, recall, auc-roc, average_precision_score
        # print(f"Test Loss: {test_loss.item()}")
            with open('results_wo_twitter.txt', 'a') as f:
                f.write(f"AUC: {auc}\n")
                f.write(f"F1 Score: {f1}\n")
                f.write(f"Precision: {precision}\n")
                f.write(f"Recall: {recall}\n")
                f.write(f"Accuracy: {accuracy}\n")
                f.write(f"Average Precision Score: {average_precision}\n")
                
                f.write('\n')

if __name__ == '__main__':
    main()