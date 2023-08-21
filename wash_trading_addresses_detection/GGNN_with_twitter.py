#!/usr/bin/env python
# coding: utf-8

import dgl
import torch
import torch.nn.functional as F
from dgl.nn.pytorch import GatedGraphConv
from sklearn.metrics import roc_auc_score
import numpy as np
import torch.backends.cudnn as cudnn
import random
import pickle as pkl

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

with open('G_train_dgl_twitter.gpickle', 'rb') as f:
    G_train_dgl_with_twitter_features = pkl.load(f)
    
with open('G_val_dgl_twitter.gpickle', 'rb') as f:
    G_val_dgl_with_twitter_features = pkl.load(f)
    
with open('G_test_dgl_twitter.gpickle', 'rb') as f:
    G_test_dgl_with_twitter_features = pkl.load(f)



hidden_dim = 128
num_classes = 2
n_steps = 5
n_etypes = 1


# Get the number of input features
in_dim = G_train_dgl_with_twitter_features.ndata['ethereum_twitter_combined_features'].shape[1]


def main():
    for i in range(10):
        # Set the random seed, a randamly selected number
        seed = random.randint(0, 1000)
        print(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = GGSNNModel(in_dim, hidden_dim, num_classes, n_steps, n_etypes)
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

            labels = G_train_dgl_with_twitter_features.ndata['label'].squeeze()
            features = G_train_dgl_with_twitter_features.ndata['ethereum_twitter_ethereum_twitter_combined_features']

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
            subgraph = dgl.node_subgraph(G_train_dgl_with_twitter_features, selected_indices)

            # Get the selected features and labels
            selected_features = subgraph.ndata['ethereum_twitter_combined_features']
            selected_labels = subgraph.ndata['label'].squeeze()

            # Forward pass and compute the loss
            logits = model(subgraph, selected_features.float())
            labels = F.one_hot(selected_labels, num_classes=2).float()
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                # Create balanced validation set
                labels = G_val_dgl_with_twitter_features.ndata['label'].squeeze()

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
                subgraph = dgl.node_subgraph(G_val_dgl_with_twitter_features, selected_indices)

                # Get the selected features and labels
                selected_features = subgraph.ndata['ethereum_twitter_combined_features']
                selected_labels = subgraph.ndata['label'].squeeze()

                # Validation
                logits = model(subgraph, selected_features.float())
                labels = F.one_hot(selected_labels, num_classes=2).float()
                val_loss = criterion(logits, labels)
                
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(model)
                torch.save(model.state_dict(), 'best_model.pt')
            print(f"Epoch: {epoch + 1}/{num_epochs}, Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}")

        best_model.eval()
        with torch.no_grad():
            # Create balanced testing set
            labels = G_test_dgl_with_twitter_features.ndata['label'].squeeze()

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
            subgraph = dgl.node_subgraph(G_test_dgl_with_twitter_features, selected_indices)

            # Get the selected features and labels
            selected_features = subgraph.ndata['ethereum_twitter_combined_features']
            ground_truth = subgraph.ndata['label'].squeeze()

            # Testing
            logits = best_model(subgraph, selected_features.float())
            _, predicted_labels = torch.max(logits, 1)

            # Calculate additional evaluation metrics for testing
            predicted_probs = F.softmax(logits, dim=1)[:, 1]
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
            with open("GGNN_with_results.txt", "a") as f:
                # need to write random seed, validation loss, test loss, auc, f1, precision, recall
                f.write(f"Random seed: {seed}, Epoch: {epoch + 1}/{num_epochs}, Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}, Macro-F1: {macro_f1:.4f}, Macro-Precision: {macro_precision:.4f}, Macro-recall: {macro_recall:.4f}\n")
            print(f"AUC: {auc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}, Macro-F1: {macro_f1:.4f}, Macro-Precision: {macro_precision:.4f}, Macro-recall: {macro_recall:.4f}\n")

if __name__ == '__main__':
    main()