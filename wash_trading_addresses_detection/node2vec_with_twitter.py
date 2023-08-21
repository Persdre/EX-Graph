import torch
import torch_geometric
import networkx as nx
import pandas as pd
import numpy as np
import pickle as pkl
import csv
from torch_geometric.nn import Node2Vec
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning
import warnings
import re
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from gensim.models import Word2Vec
from node2vec import Node2Vec

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def string_to_float_list(s):
    numbers = re.findall(r"[-+]?\d*\.\d+e?[-+]?\d*|\d+", s)
    return [float(num) for num in numbers]

def generate_node2vec_embeddings(graph, dimensions=128, walk_length=10, num_walks=10, workers=4):
    node2vec = Node2Vec(graph, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers)
    model = node2vec.fit(window=10, min_count=1, workers=workers)
    return {node: model.wv[node] for node in graph.nodes()}

def main():
    # Here, you'll need to define or load your graph G
    # As a placeholder:
    # G = nx.read_edgelist('your_edgelist.txt')

    # Generate the Node2Vec embeddings
    node2vec_embeddings = generate_node2vec_embeddings(G)

    # Load your additional data
    with open('address_to_dgl_node.pkl', 'rb') as f:
        address_to_dgl_node = pkl.load(f)

    with open('train_positive.pkl', 'rb') as f:
        train_positive = pkl.load(f)

    with open('train_negative.pkl', 'rb') as f:
        train_negative = pkl.load(f)

    with open('test_positive.pkl', 'rb') as f:
        test_positive = pkl.load(f)

    with open('test_negative.pkl', 'rb') as f:
        test_negative = pkl.load(f)

    df = pd.read_csv('twitter_wallet_address_all_features_with_node_id_pca.csv')
    df_combined_node_wallet_values = pd.read_csv('combined_node_wallet_values.csv')


    train_embeddings, train_labels = [], []
    test_embeddings, test_labels = [], []

    for nodes_list, embeddings_list, labels_list, label in [(train_positive, train_embeddings, train_labels, 1), 
                                                           (train_negative, train_embeddings, train_labels, 0),
                                                           (test_positive, test_embeddings, test_labels, 1), 
                                                           (test_negative, test_embeddings, test_labels, 0)]:
        for node in nodes_list:
            # Using Node2Vec embeddings
            base_embedding = node2vec_embeddings.get(node, np.zeros(128))
            
            # Rest of your logic to combine with Twitter embeddings...
            # Note: Ensure your logic here provides a value for twitter_deepwalk_embedding and twitter_semantic_embedding.
            twitter_embedding = np.concatenate((base_embedding, twitter_deepwalk_embedding, twitter_semantic_embedding))
            
            embeddings_list.append(twitter_embedding)
            labels_list.append(label)

    # Convert lists to arrays for sklearn
    train_embeddings = np.array(train_embeddings)
    train_labels = np.array(train_labels)
    test_embeddings = np.array(test_embeddings)
    test_labels = np.array(test_labels)

    # Define and train the classifier
    clf = LogisticRegression(random_state=0, max_iter=3000)
    clf.fit(train_embeddings, train_labels)

    # Predict and evaluate
    test_predictions = clf.predict(test_embeddings)

    auc = roc_auc_score(test_labels, test_predictions)
    f1 = f1_score(test_labels, test_predictions)
    precision = precision_score(test_labels, test_predictions)
    recall = recall_score(test_labels, test_predictions)

    print(f'auc: {auc}')
    print(f'f1: {f1}')
    print(f'precision: {precision}')
    print(f'recall: {recall}')

if __name__ == "__main__":
    main()