#!/usr/bin/env python
# coding: utf-8

import dgl
import torch
import torch.nn as nn
from dgl.nn.pytorch import GATConv
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
import pickle as pkl
import copy

class GATModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, dropout_rate=0):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, num_heads=num_heads)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, num_heads)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.conv3 = GATConv(hidden_dim * num_heads, out_dim, num_heads)

    def forward(self, g, h):
        h = self.conv1(g, h).flatten(1)
        h = self.conv2(g, h).flatten(1)
        h = self.conv3(g, h).mean(1)
        return h

def load_data(file_path):
    with open(file_path, 'rb') as f:
        return pkl.load(f)

def generate_edge_embeddings(h, edges):
    src, dst = edges[0], edges[1]
    return torch.cat([h[src], h[dst]], dim=1)

def train_one_epoch(model, optimizer, criterion, graph, features, pos_indices, neg_indices):
    model.train()

    logits = model(graph, features)
    pos_train_edge_embs = generate_edge_embeddings(logits, pos_indices)
    neg_train_edge_embs = generate_edge_embeddings(logits, neg_indices)

    train_edge_embs = torch.cat([pos_train_edge_embs, neg_train_edge_embs], dim=0)
    train_edge_labels = torch.cat([torch.ones(pos_train_edge_embs.shape[0]), torch.zeros(neg_train_edge_embs.shape[0])], dim=0).unsqueeze(1)

    transform = nn.Sequential(
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    )

    loss = criterion(transform(train_edge_embs), train_edge_labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def evaluate_model(model, graph, features, pos_indices, neg_indices, criterion, transform):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        pos_val_edge_embs = generate_edge_embeddings(logits, pos_indices)
        neg_val_edge_embs = generate_edge_embeddings(logits, neg_indices)

        val_edge_embs = torch.cat([pos_val_edge_embs, neg_val_edge_embs], dim=0)
        val_edge_labels = torch.cat([torch.ones(pos_val_edge_embs.shape[0]), torch.zeros(neg_val_edge_embs.shape[0])], dim=0).unsqueeze(1)

        val_loss = criterion(transform(val_edge_embs), val_edge_labels)

    return val_loss.item()

def test_model(model, graph, features, pos_indices, neg_indices, transform):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)

        pos_test_edge_embs = generate_edge_embeddings(logits, pos_indices)
        neg_test_edge_embs = generate_edge_embeddings(logits, neg_indices)
        test_edge_embs = torch.cat([pos_test_edge_embs, neg_test_edge_embs], dim=0)
        test_edge_labels = torch.cat([torch.ones(pos_test_edge_embs.shape[0]), torch.zeros(neg_test_edge_embs.shape[0])], dim=0)

        predictions = torch.sigmoid(transform(test_edge_embs))
        predictions = predictions.view(-1).cpu().numpy()
        test_edge_labels = test_edge_labels.cpu().numpy()

        auc = roc_auc_score(test_edge_labels, predictions)
        predictions_binary = (predictions > 0.5).astype(int)
        f1 = f1_score(test_edge_labels, predictions_binary)
        precision = precision_score(test_edge_labels, predictions_binary)
        recall = recall_score(test_edge_labels, predictions_binary)
        accuracy = accuracy_score(test_edge_labels, predictions_binary)

    return auc, f1, precision, recall, accuracy

def main():
    data_paths = [
        'positive_test_edge_indices.pkl',
        'positive_train_edge_indices.pkl',
        'positive_validation_edge_indices.pkl',
        'negative_test_edge_indices.pkl',
        'negative_train_edge_indices.pkl',
        'negative_validation_edge_indices.pkl'
    ]
    pos_test_indices, pos_train_indices, pos_val_indices, neg_test_indices, neg_train_indices, neg_val_indices = [load_data(path) for path in data_paths]

    graph_data = load_data('graph_data.pkl')
    graph, features, labels = graph_data['graph'], graph_data['features'], graph_data['labels']

    num_epochs = 200
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GATModel(features.shape[1], 128, 128, 8).to(device)
    features = features.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss().to(device)
    transform = nn.Sequential(
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    ).to(device)

    best_model = None
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        loss = train_one_epoch(model, optimizer, criterion, graph, features, pos_train_indices, neg_train_indices)
        val_loss = evaluate_model(model, graph, features, pos_val_indices, neg_val_indices, criterion, transform)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)

        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}')

    auc, f1, precision, recall, accuracy = test_model(best_model, graph, features, pos_test_indices, neg_test_indices, transform)

    print(f'Test Results - AUC: {auc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}')

if __name__ == '__main__':
    main()
