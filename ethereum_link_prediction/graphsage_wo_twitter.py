#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning
import warnings
import copy
import numpy as np
import pickle
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv
from sklearn.model_selection import ParameterGrid
from torch.optim import Adam


class Model(torch.nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, dropout_rate=0.1):
        super(Model, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, aggregator_type='mean')
        self.conv2 = SAGEConv(h_feats, h_feats, aggregator_type='mean')  # Added one more layer
        self.conv3 = SAGEConv(h_feats, num_classes, aggregator_type='mean')
        self.dropout = nn.Dropout(dropout_rate)  # Dropout layer
        self.batchnorm = nn.BatchNorm1d(h_feats)  # Batch Normalization layer

    def forward(self, graph, x):
        h = self.conv1(graph, x)
        h = F.relu(h)
        h = self.dropout(h)  # Apply dropout
        h = self.batchnorm(h)  # Apply batch normalization
        h = self.conv2(graph, h)
        h = F.relu(h)
        h = self.dropout(h)  # Apply dropout
        h = self.conv3(graph, h)
        return h


# read G_dgl graph
with open('ethereum_with_twitter_features.pkl', 'rb') as f:
    G_dgl = pickle.load(f)

# for only structure features
model = Model(8, 128, 128, 0.1)


# train on positive edges, negative edges; also use validation edges to stop epochs
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()



best_val_loss = float('inf')
best_model = None
num_epochs = 200
patience = 20
early_stopping_counter = 0

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

def generate_edge_embeddings(h, edges):
    # Extract the source and target node indices from the edges
    src, dst = edges[0], edges[1]
    
    # Use the node indices to get the corresponding node embeddings
    src_embed = h[src]
    dst_embed = h[dst]

    # Concatenate the source and target node embeddings
    edge_embs = torch.cat([src_embed, dst_embed], dim=1)

    return edge_embs


results = pd.DataFrame(columns=['Run', 'AUC', 'F1', 'Precision', 'Recall'])
best_f1 = 0
best_auc = 0
patience = 0

def main():
    # write a loop to run 5 times and document each time's performance
    for i in range(5):
        model = Model(8, 128, 128, 0.1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()
        best_val_loss = float('inf')
        best_model_wts = copy.deepcopy(model.state_dict())
        
        linear = nn.Linear(256, 1)

        for epoch in range(num_epochs):
            model.train()
            
            # forward pass
            logits = model(G_dgl, G_dgl.ndata['combined_features'])
            
            # generate edge embeddings
            pos_train_edge_embs = generate_edge_embeddings(logits, positive_train_edge_indices)
            neg_train_edge_embs = generate_edge_embeddings(logits, negative_train_edge_indices)
            
            # concatenete positive and negative edge embeddings
            train_edge_embs = torch.cat([pos_train_edge_embs, neg_train_edge_embs], dim=0)
            train_edge_labels = torch.cat([torch.ones(pos_train_edge_embs.shape[0]), torch.zeros(neg_train_edge_embs.shape[0])], dim=0).unsqueeze(1)
            
            # calculate loss
            loss = criterion(linear(train_edge_embs), train_edge_labels)
            print(f"Training Loss: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
            # validation
            model.eval()
            
            with torch.no_grad():
                # repeat the same process as above for validation samples
                logits = model(G_dgl, G_dgl.ndata['combined_features'].float())
                pos_val_edge_embs = generate_edge_embeddings(logits, positive_validation_edge_indices)
                neg_val_edge_embs = generate_edge_embeddings(logits, negative_validation_edge_indices)
                val_edge_embs = torch.cat([pos_val_edge_embs, neg_val_edge_embs], dim=0)
                val_edge_labels = torch.cat([torch.ones(pos_val_edge_embs.shape[0]), torch.zeros(neg_val_edge_embs.shape[0])], dim=0).unsqueeze(1)
                
                val_loss = criterion(linear(val_edge_embs), val_edge_labels)
                print(f"Validation Loss: {val_loss.item()}")
                
                # early stopping based on validation loss
                if val_loss <= best_val_loss:
                    best_val_loss = val_loss
                    patience = 0
                    best_model_wts = copy.deepcopy(model.state_dict())
                else:
                    patience += 1
                    if patience == 10:
                        print('early stopping due to validation loss not improving')
                        break
        
        # Load best model weights
        model.load_state_dict(best_model_wts)

        # switch to evaluation mode
        model.eval()

        with torch.no_grad():
            # generate the embeddings using the best model
            logits = model(G_dgl, G_dgl.ndata['combined_features'].float())

            # generate edge embeddings for the test samples
            pos_test_edge_embs = generate_edge_embeddings(logits, positive_test_edge_indices)
            neg_test_edge_embs = generate_edge_embeddings(logits, negative_test_edge_indices)

            # concatenate the positive and negative edge embeddings and labels
            test_edge_embs = torch.cat([pos_test_edge_embs, neg_test_edge_embs], dim=0)
            test_edge_labels = torch.cat([torch.ones(pos_test_edge_embs.shape[0]), torch.zeros(neg_test_edge_embs.shape[0])], dim=0)

            # calculate predictions using the linear layer
            predictions = torch.sigmoid(linear(test_edge_embs))
            
            # reshape the predictions and the labels
            predictions = predictions.view(-1).cpu().numpy()
            test_edge_labels = test_edge_labels.cpu().numpy()

            # calculate scores
            auc = roc_auc_score(test_edge_labels, predictions)
            predictions_binary = (predictions > 0.5).astype(int)
            f1 = f1_score(test_edge_labels, predictions_binary)
            precision = precision_score(test_edge_labels, predictions_binary)
            recall = recall_score(test_edge_labels, predictions_binary)

        print(f"AUC: {auc}")
        print(f"F1 Score: {f1}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        
        # append the results to the DataFrame
        results = results.append({
            'Run': i + 1,
            'AUC': auc,
            'F1': f1,
            'Precision': precision,
            'Recall': recall
        }, ignore_index=True)

        # save the best model
        torch.save(best_model_wts, f"best_model_run_{i + 1}.pth")

    # Save the results to a CSV file
    results.to_csv('results.csv', index=False)

if __name__ == '__main__':
    main()