#!/usr/bin/env python
# coding: utf-8

import pickle as pkl
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score

def load_data():
    with open('positive_test_edge_indices.pkl', 'rb') as f:
        positive_test_edge_indices = pkl.load(f)
    
    with open('positive_train_edge_indices.pkl', 'rb') as f:
        positive_train_edge_indices = pkl.load(f)
    
    with open('positive_validation_edge_indices.pkl', 'rb') as f:
        positive_validation_edge_indices = pkl.load(f)
    
    with open('negative_test_edge_indices.pkl', 'rb') as f:
        negative_test_edge_indices = pkl.load(f)
    
    with open('negative_train_edge_indices.pkl', 'rb') as f:
        negative_train_edge_indices = pkl.load(f)
    
    with open('negative_validation_edge_indices.pkl', 'rb') as f:
        negative_validation_edge_indices = pkl.load(f)
    
    with open('deepwalk_embeddings_for_matching_lp_1.pkl', 'rb') as f:
        embeddings = pkl.load(f)
    
    return positive_test_edge_indices, positive_train_edge_indices, positive_validation_edge_indices, negative_test_edge_indices, negative_train_edge_indices, negative_validation_edge_indices, embeddings

def generate_edge_embeddings(h, edges):
    src, dst = edges[0], edges[1]
    src_embed = h[src]
    dst_embed = h[dst]
    edge_embs = torch.cat([src_embed, dst_embed], dim=1)
    return edge_embs.detach().cpu()

def train_classifier(train_edge_embeddings, train_edge_labels):
    return LogisticRegression().fit(train_edge_embeddings, train_edge_labels)

def evaluate_classifier(clf, test_edge_embeddings, test_edge_labels):
    y_pred = clf.predict(test_edge_embeddings)
    return {
        'auc-roc': roc_auc_score(test_edge_labels, y_pred),
        'f1': f1_score(test_edge_labels, y_pred),
        'precision': precision_score(test_edge_labels, y_pred),
        'recall': recall_score(test_edge_labels, y_pred),
        'accuracy': accuracy_score(test_edge_labels, y_pred)
    }

def main():
    data = load_data()
    pos_train_edge_embeddings = generate_edge_embeddings(data[6], data[1])
    neg_train_edge_embeddings = generate_edge_embeddings(data[6], data[4])
    train_edge_embeddings = torch.cat([pos_train_edge_embeddings, neg_train_edge_embeddings], dim=0)
    train_edge_labels = torch.cat([torch.ones(pos_train_edge_embeddings.shape[0]), torch.zeros(neg_train_edge_embeddings.shape[0])], dim=0)
    
    clf = train_classifier(train_edge_embeddings, train_edge_labels)
    
    pos_test_edge_embeddings = generate_edge_embeddings(data[6], data[0])
    neg_test_edge_embeddings = generate_edge_embeddings(data[6], data[3])
    test_edge_embeddings = torch.cat([pos_test_edge_embeddings, neg_test_edge_embeddings], dim=0)
    test_edge_labels = torch.cat([torch.ones(pos_test_edge_embeddings.shape[0]), torch.zeros(neg_test_edge_embeddings.shape[0])], dim=0)

    metrics = evaluate_classifier(clf, test_edge_embeddings, test_edge_labels)
    for key, value in metrics.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
