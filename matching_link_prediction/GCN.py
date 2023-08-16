import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import pickle as pkl
from dgl.nn import GraphConv
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
import copy

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, dropout_rate):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, hidden_size)
        self.conv3 = GraphConv(hidden_size, out_feats)
        self.dropout = nn.Dropout(dropout_rate)
        self.batchnorm1 = nn.BatchNorm1d(hidden_size)

    def forward(self, g, features):
        x = F.relu(self.conv1(g, features))
        x = self.dropout(self.batchnorm1(x))
        x = F.relu(self.conv2(g, x))
        x = self.dropout(x)
        x = self.conv3(g, x)
        return x

# Define the GCN model
model = GCN(16,128,128,0.1)

# train on positive edges, negative edges; also use validation edges to stop epochs
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

criterion = nn.BCEWithLogitsLoss()

best_val_loss = float('inf')
best_model = None
num_epochs = 200
patience = 20
early_stopping_counter = 0

with open('G_dgl_training', 'rb') as f:
    G_dgl_training = pkl.load(f)
seed = 42  # Choose a seed
# Set the seed for Python's random module
random.seed(seed)
# Set the seed for NumPy's random number generator
np.random.seed(seed)
# Set the seed for PyTorch's random number generator
torch.manual_seed(seed)

# train on positive edges, negative edges; also use validation edges to stop epochs
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

criterion = nn.BCEWithLogitsLoss()

# Define a non-linear transformation
transform = nn.Sequential(
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 1)
)

# load positive_train_edge_indices and negative_train_edge_indices, positive_validation_edge_indices and negative_validation_edge_indices, positive_test_edge_indices and negative_test_edge_indices
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

def main():
    for i in range(5):
        model = GCN(16,128,128,0.1)
        # model = model.to('cuda:1')
        # train on positive edges, negative edges; also use validation edges to stop epochs
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

if __name__ == "__main__":
    main()
