#!/usr/bin/env python
# coding: utf-8
import networkx as nx
import numpy as np
import scipy.sparse as sp
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
import dgl


# read G_train_dgl, G_val_dgl, G_test_dgl
with open("G_train_dgl.gpickle", "rb") as f:
    G_train_dgl = pickle.load(f)
    
with open("G_val_dgl.gpickle", "rb") as f:
    G_val_dgl = pickle.load(f)
    
with open("G_test_dgl.gpickle", "rb") as f:
    G_test_dgl = pickle.load(f)

# Set the random seed, a randamly selected number
class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, dropout_rate):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, hidden_size)  # added layer
        self.conv3 = GraphConv(hidden_size, out_feats)  # final layer
        self.dropout = nn.Dropout(dropout_rate)  # dropout layer
        self.batchnorm1 = nn.BatchNorm1d(hidden_size)  # batchnorm layer

    def forward(self, g, features):
        x = F.relu(self.conv1(g, features))
        x = self.dropout(x)  # apply dropout
        x = self.batchnorm1(x)  # apply batchnorm
        x = F.relu(self.conv2(g, x))
        x = self.dropout(x)  # apply dropout
        # x = self.batchnorm1(x)  # apply batchnorm
        x = self.conv3(g, x)
        return x

# Get the number of input features
in_feats = G_train_dgl.ndata['ethereum_features'].shape[1]

# Define the model hyperparameters
hidden_size = 128
out_feats = 2  # Assuming binary classification
dropout_rate = 0.1


def main():
    for i in range(10):
        # Set the random seed, a randamly selected number
        seed = random.randint(0, 1000)
        print(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # Create the GCN model
        model = GCN(in_feats, hidden_size, out_feats, dropout_rate)

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

            labels = G_train_dgl.ndata['label'].squeeze()
            features = G_train_dgl.ndata['ethereum_features']

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
            subgraph = dgl.node_subgraph(G_train_dgl, selected_indices)

            # Get the selected features and labels
            selected_features = subgraph.ndata['ethereum_features']
            selected_labels = subgraph.ndata['label'].squeeze()

            # Forward pass and compute the loss
            logits = model(subgraph, selected_features)
            labels = F.one_hot(selected_labels, num_classes=out_feats).float()
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                # Create balanced validation set
                labels = G_val_dgl.ndata['label'].squeeze()

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
                subgraph = dgl.node_subgraph(G_val_dgl, selected_indices)

                # Get the selected features and labels
                selected_features = subgraph.ndata['ethereum_features']
                selected_labels = subgraph.ndata['label'].squeeze()

                # Validation
                logits = model(subgraph, selected_features)
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
            labels = G_test_dgl.ndata['label'].squeeze()

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
            subgraph = dgl.node_subgraph(G_test_dgl, selected_indices)

            # Get the selected features and labels
            selected_features = subgraph.ndata['ethereum_features']
            ground_truth = subgraph.ndata['label'].squeeze()

            # Testing
            logits = best_model(subgraph, selected_features)
            _, predicted_labels = torch.max(logits, 1)

            # Calculate additional evaluation metrics for testing
            predicted_probs = F.softmax(logits, dim=1)[:, 1]
            # change the threshold to 0.5
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
            with open("GCN_wo_results.txt", "a") as f:
                # need to write random seed, validation loss, test loss, auc, f1, precision, recall
                f.write(f"Random seed: {seed}, Epoch: {epoch + 1}/{num_epochs}, Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}, Macro-F1: {macro_f1:.4f}, Macro-Precision: {macro_precision:.4f}, Macro-recall: {macro_recall:.4f}\n")
            print(f"AUC: {auc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}, Macro-F1: {macro_f1:.4f}, Macro-Precision: {macro_precision:.4f}, Macro-recall: {macro_recall:.4f}\n")

if __name__ == '__main__':
    main()