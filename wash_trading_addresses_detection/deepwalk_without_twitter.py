import torch
import torch_geometric
import networkx as nx
import pandas as pd
import numpy as np
import pickle as pkl
import csv
from torch_geometric.nn import Node2Vec
from torch.cuda.amp import GradScaler, autocast
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning
import warnings
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.optim as optim
import re

def string_to_float_list(s):
    # Use regex to extract numbers from the string
    numbers = re.findall(r"[-+]?\d*\.\d+e?[-+]?\d*|\d+", s)
    return [float(num) for num in numbers]



# use node embeddings, fit into the clf evaluate, get auc f1 precision recall accuracy macro-f1
# load train_positive.pkl and train_negative.pkl
with open('deepwalk_embeddings_0.pkl', 'rb') as f:
    embeddings = pkl.load(f)
    
with open('address_to_dgl_node.pkl', 'rb') as f:
    address_to_dgl_node = pkl.load(f)

with open('train_positive.pkl', 'rb') as f:
    train_positive = pkl.load(f)
    
with open('train_negative.pkl', 'rb') as f:
    train_negative = pkl.load(f)
    
# load test_positive.pkl and test_negative.pkl
with open('test_positive.pkl', 'rb') as f:
    test_positive = pkl.load(f)
    
with open('test_negative.pkl', 'rb') as f:
    test_negative = pkl.load(f)
    
def main():
    # Initialize lists to store embeddings and labels
    train_positive_embeddings = []
    train_negative_embeddings = []
    train_nodes_labels = []

    test_positive_embeddings = []
    test_negative_embeddings = []
    test_nodes_labels = []

    # Assuming 'embeddings' holds your precomputed node embeddings and 'address_to_dgl_node' maps addresses to node IDs
    for node in train_positive:
        train_positive_embeddings.append(embeddings[address_to_dgl_node[node]].detach().cpu().numpy())

    for node in train_negative:
        train_negative_embeddings.append(embeddings[address_to_dgl_node[node]].detach().cpu().numpy())

    # Combine positive and negative embeddings
    train_nodes_embeddings = train_positive_embeddings + train_negative_embeddings

    # Create corresponding labels
    train_nodes_labels = [1] * len(train_positive_embeddings) + [0] * len(train_negative_embeddings)

    for node in test_positive:
        test_positive_embeddings.append(embeddings[address_to_dgl_node[node]].detach().cpu().numpy())
        
    for node in test_negative:
        test_negative_embeddings.append(embeddings[address_to_dgl_node[node]].detach().cpu().numpy())
        
    # Combine positive and negative embeddings
    test_nodes_embeddings = test_positive_embeddings + test_negative_embeddings

    # Create corresponding labels
    test_nodes_labels = [1] * len(test_positive_embeddings) + [0] * len(test_negative_embeddings)

    # define,fit and predict the clf
    clf = LogisticRegression(random_state=0, max_iter=3000)
    clf.fit(train_nodes_embeddings, train_nodes_labels)
    test_nodes_predictions = clf.predict(test_nodes_embeddings)

    # Hyperparameters
    input_size = 128
    hidden_size = 256
    num_classes = 1
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 100

    # Convert NumPy arrays to PyTorch tensors
    input_tensor = torch.tensor(np.array(train_nodes_embeddings), dtype=torch.float32)
    label_tensor = torch.tensor(np.array(train_nodes_labels).reshape((len(train_nodes_labels),1)), dtype=torch.float32)

    # Create a DataLoader for batch training
    dataset = torch.utils.data.TensorDataset(input_tensor, label_tensor)
    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    # Instantiate the MLP
    model = MLP(input_size, hidden_size, num_classes)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')


    # Convert test data to PyTorch tensors
    test_input_tensor = torch.tensor(np.array(test_nodes_embeddings), dtype=torch.float32)

    # Evaluate the model on test data
    model.eval()
    with torch.no_grad():
        test_nodes_predictions = model(test_input_tensor).numpy()
        test_nodes_predictions = test_nodes_predictions.reshape(len(test_nodes_predictions)).tolist()

    test_nodes_predictions = [1 if value > 0.5 else 0 for value in test_nodes_predictions]
    auc = roc_auc_score(test_nodes_labels, test_nodes_predictions)
    f1 = f1_score(test_nodes_labels, test_nodes_predictions)
    precision = precision_score(test_nodes_labels, test_nodes_predictions)
    recall = recall_score(test_nodes_labels, test_nodes_predictions)

    print("without twitter feature")
    print(f'auc: {auc}')
    print(f'f1: {f1}')
    print(f'precision: {precision}')
    print(f'recall: {recall}')


    # Initialize lists to store embeddings and labels
    train_positive_embeddings = []
    train_negative_embeddings = []
    train_nodes_labels = []

    test_positive_embeddings = []
    test_negative_embeddings = []
    test_nodes_labels = []
    # for all_features column, use pca to normalize it to 8 dimensions
    # Initialize PCA and specify the number of components (dimensions) you want
    # Convert the strings to actual lists and then to numpy arrays
    df['all_features'] = df['all_features'].apply(string_to_float_list)
    df['all_features'] = df['all_features'].apply(np.array)

    # Convert the entire 'all_features' column into a 2D array
    data_matrix = np.array(df['all_features'].tolist())

    # Initialize and fit PCA on the entire data
    pca = PCA(n_components=8)
    transformed_data = pca.fit_transform(data_matrix)

    # If you wish, you can convert the transformed data back into the DataFrame
    df['8_dimension_semantic_features'] = list(transformed_data)

    for node in train_positive:
        # Twitter deepwalk embedding
        if node in df_combined_node_wallet_values['Wallet_Address'].values:
            twitter_deepwalk_embedding = df_combined_node_wallet_values[df_combined_node_wallet_values['Wallet_Address'] == node]['values'].values[0]
            float_list = string_to_float_list(twitter_deepwalk_embedding)
            twitter_deepwalk_embedding = np.array(float_list)
        else:
            twitter_deepwalk_embedding = np.zeros(8)
        
            
        if node in df['Wallet_Address'].values:
            twitter_semantic_embedding = df[df['Wallet_Address'] == node]['8_dimension_semantic_features'].values[0]
        else:
            twitter_semantic_embedding = np.zeros(8)
        
        # combine all embeddings
        twitter_embedding = np.concatenate((twitter_deepwalk_embedding, twitter_semantic_embedding))
        node_embedding = np.concatenate((embeddings[address_to_dgl_node[node]].detach().cpu().numpy(), twitter_embedding))
        train_positive_embeddings.append(node_embedding)

    for node in train_negative:
        # Twitter deepwalk embedding
        if node in df_combined_node_wallet_values['Wallet_Address'].values:
            twitter_deepwalk_embedding = df_combined_node_wallet_values[df_combined_node_wallet_values['Wallet_Address'] == node]['values'].values[0]
            float_list = string_to_float_list(twitter_deepwalk_embedding)
            twitter_deepwalk_embedding = np.array(float_list)
        else:
            twitter_deepwalk_embedding = np.zeros(8)
        # print(len(twitter_deepwalk_embedding))
        
        # if(len(twitter_deepwalk_embedding) != 8):
        #     print(node)
        #     print(twitter_deepwalk_embedding)
            
        if node in df['Wallet_Address'].values:
            twitter_semantic_embedding = df[df['Wallet_Address'] == node]['8_dimension_semantic_features'].values[0]
        else:
            twitter_semantic_embedding = np.zeros(8)
        
        # combine all embeddings
        twitter_embedding = np.concatenate((twitter_deepwalk_embedding, twitter_semantic_embedding))
        node_embedding = np.concatenate((embeddings[address_to_dgl_node[node]].detach().cpu().numpy(), twitter_embedding))
        train_negative_embeddings.append(node_embedding)
        
    
    # Combine positive and negative embeddings
    train_nodes_embeddings = np.concatenate((train_positive_embeddings, train_negative_embeddings))
    # Create corresponding labels
    train_nodes_labels = [1] * len(train_positive_embeddings) + [0] * len(train_negative_embeddings)
    train_nodes_labels = np.array(train_nodes_labels)
    
    
    for node in test_positive:
        # Twitter deepwalk embedding
        if node in df_combined_node_wallet_values['Wallet_Address'].values:
            twitter_deepwalk_embedding = df_combined_node_wallet_values[df_combined_node_wallet_values['Wallet_Address'] == node]['values'].values[0]
            float_list = string_to_float_list(twitter_deepwalk_embedding)
            twitter_deepwalk_embedding = np.array(float_list)
        else:
            twitter_deepwalk_embedding = np.zeros(8)
        # print(len(twitter_deepwalk_embedding))
        
        # if(len(twitter_deepwalk_embedding) != 8):
        #     print(node)
        #     print(twitter_deepwalk_embedding)
            
        if node in df['Wallet_Address'].values:
            twitter_semantic_embedding = df[df['Wallet_Address'] == node]['8_dimension_semantic_features'].values[0]
        else:
            twitter_semantic_embedding = np.zeros(8)
        
        # combine all embeddings
        twitter_embedding = np.concatenate((twitter_deepwalk_embedding, twitter_semantic_embedding))
        node_embedding = np.concatenate((embeddings[address_to_dgl_node[node]].detach().cpu().numpy(), twitter_embedding))
        test_positive_embeddings.append(node_embedding)

    for node in test_negative:
        # Twitter deepwalk embedding
        if node in df_combined_node_wallet_values['Wallet_Address'].values:
            twitter_deepwalk_embedding = df_combined_node_wallet_values[df_combined_node_wallet_values['Wallet_Address'] == node]['values'].values[0]
            float_list = string_to_float_list(twitter_deepwalk_embedding)
            twitter_deepwalk_embedding = np.array(float_list)
        else:
            twitter_deepwalk_embedding = np.zeros(8)
        # print(len(twitter_deepwalk_embedding))
        
        # if(len(twitter_deepwalk_embedding) != 8):
        #     print(node)
        #     print(twitter_deepwalk_embedding)
            
        if node in df['Wallet_Address'].values:
            twitter_semantic_embedding = df[df['Wallet_Address'] == node]['8_dimension_semantic_features'].values[0]
        else:
            twitter_semantic_embedding = np.zeros(8)
        
        # combine all embeddings
        twitter_embedding = np.concatenate((twitter_deepwalk_embedding, twitter_semantic_embedding))
        node_embedding = np.concatenate((embeddings[address_to_dgl_node[node]].detach().cpu().numpy(), twitter_embedding))
        test_negative_embeddings.append(node_embedding)
    
    # Combine positive and negative embeddings
    test_nodes_embeddings = np.concatenate((train_positive_embeddings, train_negative_embeddings))
    # Create corresponding labels
    test_nodes_labels = [1] * len(train_positive_embeddings) + [0] * len(train_negative_embeddings)
    test_nodes_labels = np.array(test_nodes_labels)
        
    # define the clf
    clf = LogisticRegression(random_state=0, max_iter=3000)

    # fit the clf
    clf.fit(train_nodes_embeddings, train_nodes_labels)

    # predict the test nodes
    test_nodes_predictions = clf.predict(test_nodes_embeddings)
    auc = roc_auc_score(test_nodes_labels, test_nodes_predictions)
    f1 = f1_score(test_nodes_labels, test_nodes_predictions)
    precision = precision_score(test_nodes_labels, test_nodes_predictions)
    recall = recall_score(test_nodes_labels, test_nodes_predictions)

    print("without twitter feature")
    print(f'auc: {auc}')
    print(f'f1: {f1}')
    print(f'precision: {precision}')
    print(f'recall: {recall}')
    
if __name__ == "__main__":
    main()
