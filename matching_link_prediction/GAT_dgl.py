#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dgl
import torch
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
from sklearn.metrics import roc_auc_score
import numpy as np
import torch.backends.cudnn as cudnn
import random
import pickle as pkl
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from dgl.nn import GraphConv


# In[2]:


# change dropout rate to 0.3, 0.4, 0.5
class GATModel(torch.nn.Module):
    # hyperparameters: 0.3, 0.4, 0.5
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, dropout_rate=0):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, num_heads=num_heads)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, num_heads)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.conv3 = GATConv(hidden_dim * num_heads, out_dim, num_heads)
    
    def forward(self, g, h):
        h = self.conv1(g, h).flatten(1)
        # h = F.elu(self.dropout1(h))
        h = self.conv2(g, h).flatten(1)
        # h = F.elu(self.dropout2(h))
        h = self.conv3(g, h).mean(1)
        return h


# In[3]:


# load /home/qian/HNE/Model/GCN/Ethereum/matching_link/positive_test_edge_indices.pkl, /home/qian/HNE/Model/GCN/Ethereum/matching_link/positive_train_edge_indices.pkl, /home/qian/HNE/Model/GCN/Ethereum/matching_link/positive_validation_edge_indices.pkl, /home/qian/HNE/Model/GCN/Ethereum/matching_link/negative_test_edge_indices.pkl, /home/qian/HNE/Model/GCN/Ethereum/matching_link/negative_train_edge_indices.pkl, /home/qian/HNE/Model/GCN/Ethereum/matching_link/negative_validation_edge_indices.pkl
with open('/home/qian/HNE/Model/GCN/Ethereum/matching_link/positive_test_edge_indices.pkl', 'rb') as f:
    positive_test_edge_indices = pkl.load(f)
    
with open('/home/qian/HNE/Model/GCN/Ethereum/matching_link/positive_train_edge_indices.pkl', 'rb') as f:
    positive_train_edge_indices = pkl.load(f)
    
with open('/home/qian/HNE/Model/GCN/Ethereum/matching_link/positive_validation_edge_indices.pkl', 'rb') as f:
    positive_validation_edge_indices = pkl.load(f)
    
with open('/home/qian/HNE/Model/GCN/Ethereum/matching_link/negative_test_edge_indices.pkl', 'rb') as f:
    negative_test_edge_indices = pkl.load(f)
    
with open('/home/qian/HNE/Model/GCN/Ethereum/matching_link/negative_train_edge_indices.pkl', 'rb') as f:
    negative_train_edge_indices = pkl.load(f)
    
with open('/home/qian/HNE/Model/GCN/Ethereum/matching_link/negative_validation_edge_indices.pkl', 'rb') as f:
    negative_validation_edge_indices = pkl.load(f)


# In[4]:


# print first examples of these files
# print(positive_test_edge_indices[0]))
print(positive_train_edge_indices[0])
print(negative_train_edge_indices[1])


# In[5]:


# load G_dgl_training
import dgl
with open('/home/qian/HNE/Model/GCN/Ethereum/matching_link/G_dgl_training', 'rb') as f:
    G_dgl_training = pkl.load(f)


# In[6]:


# define generate_edge_embeddings function
def generate_edge_embeddings(h, edges):
    # Extract the source and target node indices from the edges
    src, dst = edges[0], edges[1]
    
    # Use the node indices to get the corresponding node embeddings
    src_embed = h[src]
    dst_embed = h[dst]

    # Concatenate the source and target node embeddings
    edge_embs = torch.cat([src_embed, dst_embed], dim=1)

    return edge_embs


# In[7]:


# print some features examples
print(G_dgl_training.ndata['features'][0])


# In[10]:


# write a loop to run 5 times of the model and get the average performance
import copy
for i in range(5):
    model = GATModel(16,128,128,4,0)
    # Use the learning rate scheduler
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
    nn.Linear(128, 1)
    )
    
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


        # test_loss = criterion(linear(test_edge_embs), val_edge_labels)
        # calculate predictions using the linear layer
        
        predictions = torch.sigmoid(transform(test_edge_embs))
        
        # reshape the predictions and the labels
        predictions = predictions.view(-1).cpu().numpy()
        test_edge_labels = test_edge_labels.cpu().numpy()

        # calculate scores and entropyloss
        
        
        auc = roc_auc_score(test_edge_labels, predictions)
        # here use 0.5 as threshold
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
    
    # write the result to a txt file
    with open('result.txt', 'a') as f:
        # write auc, f1, precision, recall
        f.write(f"AUC: {auc}, F1 Score: {f1}, Precision: {precision}, Recall: {recall}, Accuracy: {accuracy}\n")


# 

# In[ ]:




