import torch
import torch_geometric
import networkx as nx
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning
import warnings

# Load train_positive.pkl and train_negative.pkl
with open('deepwalk_embeddings_0.pkl', 'rb') as f:
    embeddings = pkl.load(f)
    
with open('address_to_dgl_node.pkl', 'rb') as f:
    address_to_dgl_node = pkl.load(f)

with open('train_positive.pkl', 'rb') as f:
    train_positive = pkl.load(f)
    
with open('train_negative.pkl', 'rb') as f:
    train_negative = pkl.load(f)

# Load test_positive.pkl and test_negative.pkl
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
    
    for node in train_positive:
        node_embedding = embeddings[address_to_dgl_node[node]].detach().cpu().numpy()
        train_positive_embeddings.append(node_embedding)

    for node in train_negative:
        node_embedding = embeddings[address_to_dgl_node[node]].detach().cpu().numpy()
        train_negative_embeddings.append(node_embedding)
        
    # Combine positive and negative embeddings
    train_nodes_embeddings = np.concatenate((train_positive_embeddings, train_negative_embeddings))
    # Create corresponding labels
    train_nodes_labels = [1] * len(train_positive_embeddings) + [0] * len(train_negative_embeddings)
    train_nodes_labels = np.array(train_nodes_labels)
    
    for node in test_positive:
        node_embedding = embeddings[address_to_dgl_node[node]].detach().cpu().numpy()
        test_positive_embeddings.append(node_embedding)

    for node in test_negative:
        node_embedding = embeddings[address_to_dgl_node[node]].detach().cpu().numpy()
        test_negative_embeddings.append(node_embedding)
    
    # Combine positive and negative embeddings
    test_nodes_embeddings = np.concatenate((test_positive_embeddings, test_negative_embeddings))
    # Create corresponding labels
    test_nodes_labels = [1] * len(test_positive_embeddings) + [0] * len(test_negative_embeddings)
    test_nodes_labels = np.array(test_nodes_labels)
        
    # Define the clf
    clf = LogisticRegression(random_state=0, max_iter=3000)

    # Fit the clf
    clf.fit(train_nodes_embeddings, train_nodes_labels)

    # Predict the test nodes
    test_nodes_predictions = clf.predict(test_nodes_embeddings)
    auc = roc_auc_score(test_nodes_labels, test_nodes_predictions)
    f1 = f1_score(test_nodes_labels, test_nodes_predictions)
    precision = precision_score(test_nodes_labels, test_nodes_predictions)
    recall = recall_score(test_nodes_labels, test_nodes_predictions)

    print(f'auc: {auc}')
    print(f'f1: {f1}')
    print(f'precision: {precision}')
    print(f'recall: {recall}')
    
if __name__ == "__main__":
    main()
