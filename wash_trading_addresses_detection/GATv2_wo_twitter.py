#!/usr/bin/env python
# coding: utf-8

import dgl
import torch
import torch.nn.functional as F
from dgl.nn import GATv2Conv
from dgl.nn.pytorch import GATConv
from sklearn.metrics import roc_auc_score
import numpy as np
import torch.backends.cudnn as cudnn
import random
import pickle as pkl

"""
Graph Attention Networks in DGL using SPMV optimization.
References
----------
Paper: https://arxiv.org/pdf/2105.14491.pdf
Author's code: https://github.com/tech-srl/how_attentive_are_gats
"""

class GATv2(nn.Module):
    def __init__(
        self,
        num_layers,
        in_dim,
        num_hidden,
        num_classes,
        heads,
        activation,
        feat_drop,
        attn_drop,
        negative_slope,
        residual,
    ):
        super(GATv2, self).__init__()
        self.num_layers = num_layers
        self.gatv2_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gatv2_layers.append(
            GATv2Conv(
                in_dim,
                num_hidden,
                heads[0],
                feat_drop,
                attn_drop,
                negative_slope,
                False,
                self.activation,
                bias=False,
                share_weights=True,
            )
        )
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gatv2_layers.append(
                GATv2Conv(
                    num_hidden * heads[l - 1],
                    num_hidden,
                    heads[l],
                    feat_drop,
                    attn_drop,
                    negative_slope,
                    residual,
                    self.activation,
                    bias=False,
                    share_weights=True,
                )
            )
        # output projection
        self.gatv2_layers.append(
            GATv2Conv(
                num_hidden * heads[-2],
                num_classes,
                heads[-1],
                feat_drop,
                attn_drop,
                negative_slope,
                residual,
                None,
                bias=False,
                share_weights=True,
            )
        )

    def forward(self, g, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gatv2_layers[l](g, h).flatten(1)
        # output projection
        logits = self.gatv2_layers[-1](g, h).mean(1)
        return logits


# read G_train_dgl, G_val_dgl, G_test_dgl, use pickle to read
with open('G_train_dgl.gpickle', 'rb') as f:
    G_train_dgl = pkl.load(f)
    
with open('G_val_dgl.gpickle', 'rb') as f:
    G_val_dgl = pkl.load(f)
    
with open('G_test_dgl.gpickle', 'rb') as f:
    G_test_dgl = pkl.load(f)


# Get the number of input features
in_feats = G_train_dgl.ndata['ethereum_features'].shape[1]

# Define the model hyperparameters
num_layers = 3
in_dim = in_feats
num_hidden = 128
num_classes = 2
heads = [3, 3, 3]
activation = F.elu
feat_drop = 0.1
attn_drop = 0.1
negative_slope = 0.2
residual = False

# create GATv2 model
model = GATv2(num_layers, in_dim, num_hidden, num_classes, heads, activation, feat_drop, attn_drop, negative_slope, residual)

# define loss function
# Define the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()
best_val_loss = float('inf')
best_model = None
num_epochs = 200
patience = 20


def main():
    for i in range(10):
        # Set the random seed, a randamly selected number
        seed = random.randint(0, 1000)
        print(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # Create the GATv2 model
        model = GATv2(num_layers, in_dim, num_hidden, num_classes, heads, activation, feat_drop, attn_drop, 0.1, True)
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
            labels = F.one_hot(selected_labels, num_classes=2).float()
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
            # adjust threshold from 0.5 to 0.4
            predicted_labels = (predicted_probs > 0.4).float()
            auc = roc_auc_score(ground_truth.detach().numpy(), predicted_probs.detach().numpy())
            f1 = f1_score(ground_truth.detach().numpy(), predicted_labels.detach().numpy())
            precision = precision_score(ground_truth.detach().numpy(), predicted_labels.detach().numpy())
            recall = recall_score(ground_truth.detach().numpy(), predicted_labels.detach().numpy())
            accuracy = accuracy_score(ground_truth.detach().numpy(), predicted_labels.detach().numpy())
            macro_f1 = f1_score(ground_truth.detach().numpy(), predicted_labels.detach().numpy(), average='macro')
            macro_precision = precision_score(ground_truth.detach().numpy(), predicted_labels.detach().numpy(), average='macro')
            macro_recall = recall_score(ground_truth.detach().numpy(), predicted_labels.detach().numpy(), average='macro')
            # store results in a txt file
            with open("GATv2_wo_results.txt", "a") as f:
                # need to write random seed, validation loss, test loss, auc, f1, precision, recall
                f.write(f"Random seed: {seed}, Epoch: {epoch + 1}/{num_epochs}, Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}, Macro-F1: {macro_f1:.4f}, Macro-Precision: {macro_precision:.4f}, Macro-recall: {macro_recall:.4f}\n")
            print(f"AUC: {auc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}, Macro-F1: {macro_f1:.4f}, Macro-Precision: {macro_precision:.4f}, Macro-recall: {macro_recall:.4f}\n")


if __name__ == '__main__':
    main()