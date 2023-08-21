#!/usr/bin/env python
# coding: utf-8

import pickle as pkl
import torch
from torch_geometric.nn import Node2Vec
from torch.cuda.amp import GradScaler, autocast
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_data(file_name):
    """Load data from a pickle file."""
    with open(file_name, 'rb') as f:
        return pkl.load(f)


def get_embeddings_for_nodes(embeddings, nodes, address_map):
    """Retrieve embeddings for a list of nodes."""
    return [embeddings[address_map[node]].detach().cpu().numpy() for node in nodes]


def train_model():
    """Train the Node2Vec model."""
    scaler = GradScaler()
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        with autocast():
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)


def main():
    global model, loader, optimizer  # Making these global for the train_model function. Ideally, these should be passed as arguments to train_model.

    # Load the graph
    G_LP_connected_dgl = load_data('G_LP_connected_dgl.pkl')
    src, dst = G_LP_connected_dgl.edges()
    edge_index = torch.tensor([src.tolist(), dst.tolist()], dtype=torch.long).contiguous().to(device)

    # Train Node2Vec model and store embeddings
    for i in range(5):
        print(f'training round: {i}')
        model = Node2Vec(edge_index, embedding_dim=128, walk_length=20, context_size=5, 
                         walks_per_node=40, num_negative_samples=1, p=0.5, q=2, sparse=True).to(device)
        loader = model.loader(batch_size=128, shuffle=True, num_workers=4)
        optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)

        for epoch in range(1, 101):
            loss = train_model()
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')

        print(f'finish training round: {i}')
        embeddings = model(torch.arange(edge_index.max().item() + 1).to(device))
        with open(f'node2vec_embeddings_{i + 1}.pkl', 'wb') as f:
            pkl.dump(embeddings, f)
        print(f'finish saving embeddings: {i}')

    # Load embeddings and training/testing data
    node2vec_embeddings_0 = load_data('node2vec_embeddings_0.pkl')
    train_positive = load_data('train_positive.pkl')
    train_negative = load_data('train_negative.pkl')
    test_positive = load_data('test_positive.pkl')
    test_negative = load_data('test_negative.pkl')
    address_to_dgl_node = load_data('address_to_dgl_node.pkl')

    # Extract embeddings for training and test nodes
    train_positive_embeddings = get_embeddings_for_nodes(node2vec_embeddings_0, train_positive, address_to_dgl_node)
    train_negative_embeddings = get_embeddings_for_nodes(node2vec_embeddings_0, train_negative, address_to_dgl_node)
    train_embeddings = train_positive_embeddings + train_negative_embeddings
    train_labels = [1] * len(train_positive_embeddings) + [0] * len(train_negative_embeddings)

    test_positive_embeddings = get_embeddings_for_nodes(node2vec_embeddings_0, test_positive, address_to_dgl_node)
    test_negative_embeddings = get_embeddings_for_nodes(node2vec_embeddings_0, test_negative, address_to_dgl_node)
    test_embeddings = test_positive_embeddings + test_negative_embeddings
    test_labels = [1] * len(test_positive_embeddings) + [0] * len(test_negative_embeddings)

    # Train a Logistic Regression classifier
    clf = LogisticRegression(random_state=0, max_iter=3000)
    clf.fit(train_embeddings, train_labels)

    # Evaluate the classifier
    test_predictions = clf.predict(test_embeddings)
    auc = roc_auc_score(test_labels, test_predictions)
    f1 = f1_score(test_labels, test_predictions)
    precision = precision_score(test_labels, test_predictions)
    recall = recall_score(test_labels, test_predictions)
    accuracy = (test_predictions == test_labels).mean()
    macro_f1 = f1_score(test_labels, test_predictions, average='macro')

    # Print evaluation metrics
    print(f'auc: {auc}')
    print(f'f1: {f1}')
    print(f'precision: {precision}')
    print(f'recall: {recall}')
    print(f'accuracy: {accuracy}')
    print(f'macro_f1: {macro_f1}')


if __name__ == "__main__":
    main()
