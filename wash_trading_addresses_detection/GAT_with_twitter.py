#!/usr/bin/env python
# coding: utf-8
import dgl
import torch
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
from sklearn.metrics import roc_auc_score
import numpy as np
import torch.backends.cudnn as cudnn
import random
import pickle as pkl
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import copy
import random

# read G_train_dgl, G_val_dgl, G_test_dgl, use pickle to read
with open('G_train_dgl_twitter.gpickle', 'rb') as f:
    G_train_dgl_twitter= pkl.load(f)
    
with open('G_val_dgl_twitter.gpickle', 'rb') as f:
    G_val_dgl_twitter = pkl.load(f)
    
with open('G_test_dgl_twitter.gpickle', 'rb') as f:
    G_test_dgl_twitter = pkl.load(f)


class GATModel(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, dropout_rate=0.1):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, num_heads=num_heads)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, num_heads)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.conv3 = GATConv(hidden_dim * num_heads, out_dim, num_heads)
    
    def forward(self, g, h):
        h = self.conv1(g, h).flatten(1)
        h = F.elu(self.dropout1(h))
        h = self.conv2(g, h).flatten(1)
        h = F.elu(self.dropout2(h))
        h = self.conv3(g, h).mean(1)
        return h


# Get the number of input features
in_feats = G_train_dgl_twitter.ndata['ethereum_twitter_combined_features'].shape[1]

# Define the model hyperparameters
hidden_size = 128
out_feats = 2  # Assuming binary classification
num_heads = 3

# Create the GCN model
model = GATModel(in_feats, hidden_size, out_feats, num_heads)


# Define the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()
best_val_loss = float('inf')
best_model = None
num_epochs = 200
patience = 20

# Get the number of input features
in_feats = G_train_dgl_twitter.ndata['ethereum_twitter_combined_features'].shape[1]

# Define the model hyperparameters
hidden_size = 128
out_feats = 2  # Assuming binary classification
num_heads = 3


# for G_train, G_test, G_val, add self loop
G_train_dgl_twitter = dgl.add_self_loop(G_train_dgl_twitter)
G_val_dgl_twitter = dgl.add_self_loop(G_val_dgl_twitter)
G_test_dgl_twitter = dgl.add_self_loop(G_test_dgl_twitter)


def main():
    for i in range(10):
        # Set the random seed, a randamly selected number
        seed = random.randint(0, 1000)
        print(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # Create the GAT model
        model = GATModel(24, 128, 2, 3)
        # Define the optimizer and loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()
        best_val_loss = float('inf')
        best_model = None
        num_epochs = 200
        patience = 20

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()

            labels = G_train_dgl_twitter.ndata['label'].squeeze()
            features = G_train_dgl_twitter.ndata['ethereum_twitter_combined_features']

            # Select indices of 0 and 1 labels
            zero_indices = torch.where(labels == 0)[0]
            one_indices = torch.where(labels == 1)[0]
            
            # Get the minimum count between 0 and 1 labels
            min_count = min(zero_indices.shape[0], one_indices.shape[0])
            
            # Randomly select 'min_count' indices from zero_indices and one_indices each
            selected_zero_indices = zero_indices[torch.randperm(zero_indices.shape[0])[:min_count]]
            selected_one_indices = one_indices[torch.randperm(one_indices.shape[0])[:min_count]]

            # Combine the selected indices
            selected_indices = torch.cat((selected_zero_indices, selected_one_indices))

            # Shuffle the selected indices
            selected_indices = selected_indices[torch.randperm(selected_indices.shape[0])]

            # Create a subgraph from the selected indices
            subgraph = dgl.node_subgraph(G_train_dgl_twitter, selected_indices)

            # Get the selected features and labels
            selected_features = subgraph.ndata['ethereum_twitter_combined_features']
            selected_labels = subgraph.ndata['label'].squeeze()

            # Forward pass and compute the loss
            logits = model(subgraph, selected_features.float())
            labels = F.one_hot(selected_labels, num_classes=out_feats).float()
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                # Create balanced validation set
                labels = G_val_dgl_twitter.ndata['label'].squeeze()

                # Select indices of 0 and 1 labels
                zero_indices = torch.where(labels == 0)[0]
                one_indices = torch.where(labels == 1)[0]

                # Get the minimum count between 0 and 1 labels
                min_count = min(zero_indices.shape[0], one_indices.shape[0])

                # Randomly select 'min_count' indices from zero_indices and one_indices each
                selected_zero_indices = zero_indices[torch.randperm(zero_indices.shape[0])[:min_count]]
                selected_one_indices = one_indices[torch.randperm(one_indices.shape[0])[:min_count]]

                # Combine the selected indices
                selected_indices = torch.cat((selected_zero_indices, selected_one_indices))

                # Shuffle the selected indices
                selected_indices = selected_indices[torch.randperm(selected_indices.shape[0])]

                # Create a subgraph from the selected indices
                subgraph = dgl.node_subgraph(G_val_dgl_twitter, selected_indices)

                # Get the selected features and labels
                selected_features = subgraph.ndata['ethereum_twitter_combined_features']
                selected_labels = subgraph.ndata['label'].squeeze()

                # Validation
                logits = model(subgraph, selected_features.float())
                labels = F.one_hot(selected_labels, num_classes=out_feats).float()
            val_loss = criterion(logits, labels)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(model)
                torch.save(model.state_dict(), 'best_model.pt')
            print(f"Epoch: {epoch + 1}/{num_epochs}, Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}")            

        best_model.eval()
        with torch.no_grad():
            # Create balanced testing set
            labels = G_test_dgl_twitter.ndata['label'].squeeze()

            # Select indices of 0 and 1 labels
            zero_indices = torch.where(labels == 0)[0]
            one_indices = torch.where(labels == 1)[0]

            # Get the minimum count between 0 and 1 labels
            min_count = min(zero_indices.shape[0], one_indices.shape[0])

            # Randomly select 'min_count' indices from zero_indices and one_indices each
            selected_zero_indices = zero_indices[torch.randperm(zero_indices.shape[0])[:min_count]]
            selected_one_indices = one_indices[torch.randperm(one_indices.shape[0])[:min_count]]

            # Combine the selected indices
            selected_indices = torch.cat((selected_zero_indices, selected_one_indices))

            # Shuffle the selected indices
            selected_indices = selected_indices[torch.randperm(selected_indices.shape[0])]

            # Create a subgraph from the selected indices
            subgraph = dgl.node_subgraph(G_test_dgl_twitter, selected_indices)

            # Get the selected features and labels
            selected_features = subgraph.ndata['ethereum_twitter_combined_features']
            ground_truth = subgraph.ndata['label'].squeeze()

            # Testing
            logits = best_model(subgraph, selected_features.float())
            _, predicted_labels = torch.max(logits, 1)

            # Calculate additional evaluation metrics for testing
            predicted_probs = F.softmax(logits, dim=1)[:, 1]
            # adjust threshold here
            predicted_labels = (predicted_probs > 0.5).float()
            auc = roc_auc_score(ground_truth.detach().numpy(), predicted_probs.detach().numpy())
            f1 = f1_score(ground_truth.detach().numpy(), predicted_labels.detach().numpy())
            precision = precision_score(ground_truth.detach().numpy(), predicted_labels.detach().numpy())
            recall = recall_score(ground_truth.detach().numpy(), predicted_labels.detach().numpy())
            accuracy = accuracy_score(ground_truth.detach().numpy(), predicted_labels.detach().numpy())
            macro_f1 = f1_score(ground_truth.detach().numpy(), predicted_labels.detach().numpy(), average='macro')
            macro_precision = precision_score(ground_truth.detach().numpy(), predicted_labels.detach().numpy(), average='macro')
            macro_recall = recall_score(ground_truth.detach().numpy(), predicted_labels.detach().numpy(), average='macro')
            # store results in a txt file
            with open("GAT_with_results.txt", "a") as f:
                # need to write random seed, validation loss, test loss, auc, f1, precision, recall
                f.write(f"Random seed: {seed}, Epoch: {epoch + 1}/{num_epochs}, Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}, Macro-F1: {macro_f1:.4f}, Macro-Precision: {macro_precision:.4f}, Macro-recall: {macro_recall:.4f}\n")
            print(f"AUC: {auc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}, Macro-F1: {macro_f1:.4f}, Macro-Precision: {macro_precision:.4f}, Macro-recall: {macro_recall:.4f}\n")

if __name__ == '__main__':
    main()