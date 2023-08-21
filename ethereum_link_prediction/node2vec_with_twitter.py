import torch
import torch_geometric
import networkx as nx
import pandas as pd
import numpy as np
import pickle
import csv
from torch_geometric.nn import Node2Vec
from torch.cuda.amp import GradScaler, autocast
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning
import warnings
from sklearn.preprocessing import StandardScaler

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
mapping = {}


def main():
    with open('G_node2vec.gpickle', 'rb') as f:
        G_node2vec = pickle.load(f)

    with open('G_node2vec_reorder.gpickle', 'rb') as f:
        G_node2vec_reorder = pickle.load(f)
        
    # read G_dgl graph for later get the twitter converted features
    with open('ethereum_with_twitter_features.pkl', 'rb') as f:
        G_dgl = pickle.load(f)
        

    mapping = dict(zip(G_node2vec.nodes, G_node2vec_reorder.nodes))

    edges = np.array(G_node2vec_reorder.edges).T
    edge_index = torch.tensor(edges, dtype=torch.long).contiguous()

    def train():
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

    def get_edge_embeddings(file_path):
        edge_embs = []
        edge_labels = []

        with open(file_path, 'r') as f:
            for line in f:
                node1, node2, label = map(int, line.strip().split())
                
                # matching the node1 and node2 to the node1 and node2 in G_node2vec_reorder
                node1 = mapping[node1]
                node2 = mapping[node2]

                # features from G_node2vec_reorder
                node1_rw = embeddings[node1].detach().cpu().numpy()
                node2_rw = embeddings[node2].detach().cpu().numpy()
                
                # features from G_dgl
                node1_twitter = G_dgl.nodes[node1].data['pca_8_normalized_twitter_features'].detach().cpu().numpy().flatten()
                node2_twitter = G_dgl.nodes[node2].data['pca_8_normalized_twitter_features'].detach().cpu().numpy().flatten()
                
                
                # combine the features from twitter and G_node2vec_reorder, the length of the combined feature is 128+128=256
                node1_embedding = np.concatenate((node1_rw, node1_twitter))
                node2_embedding = np.concatenate((node2_rw, node2_twitter))

                combined_embedding = np.concatenate((node1_embedding, node2_embedding))
                edge_embs.append(combined_embedding)
                edge_labels.append(label)
        return edge_embs, edge_labels

    def clf_fit(train_edge_embs, train_edge_labels):
        scaler = StandardScaler()
        train_edge_embs = scaler.fit_transform(train_edge_embs)
        clf = LogisticRegression(max_iter=3000).fit(train_edge_embs, train_edge_labels)
        return clf, scaler

    def clf_predict(clf, scaler, test_edge_embs, test_edge_labels):
        test_edge_embs = scaler.transform(test_edge_embs)
        preds = clf.predict(test_edge_embs)
        auc = roc_auc_score(test_edge_labels, preds)
        f1 = f1_score(test_edge_labels, preds)
        precision = precision_score(test_edge_labels, preds)
        recall = recall_score(test_edge_labels, preds)
        return auc, f1, precision, recall

    def clf_evaluate(train_edge_embs, train_edge_labels, test_edge_embs, test_edge_labels):
        clf, scaler = clf_fit(train_edge_embs, train_edge_labels)
        auc, f1, precision, recall = clf_predict(clf, scaler, test_edge_embs, test_edge_labels)
        return auc, f1, precision, recall

    
    aucs, f1s, precisions, recalls = [], [], [], []
    for i in range(5):
        print(f'Start run {i}!')
        
        
        model = Node2Vec(edge_index, embedding_dim=128, walk_length=20, context_size=5, walks_per_node=10, num_negative_samples=1, p=0.5, q=2, sparse=True).to(device)
        loader = model.loader(batch_size=128, shuffle=True, num_workers=4)
        optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)
        # test 10 epochs first
        for epoch in range(1, 2):
            loss = train()
            print('Epoch: {:02d}, Loss: {:.4f}'.format(epoch, loss))
            
        print(f'finish node2vec training for run {i}!')
        
        embeddings = model(torch.arange(edge_index.max().item() + 1, device=device))

        
        
        # load embeddings
        print(f'begin to load embeddings for run {i}!')
        with open(f'embeddings_with_twitter_{i}.pkl', 'rb') as f:
            embeddings = pickle.load(f)
            
        print(f'finish loading embeddings for run {i}!')
            
        
        train_edge_embs, train_edge_labels = get_edge_embeddings('positive_train_samples.txt')
        neg_train_edge_embs, neg_train_edge_labels = get_edge_embeddings('negative_train_samples.txt')
        test_edge_embs, test_edge_labels = get_edge_embeddings('positive_test_samples.txt')
        neg_test_edge_embs, neg_test_edge_labels = get_edge_embeddings('negative_test_samples.txt')

        train_edge_embs.extend(neg_train_edge_embs)
        train_edge_labels.extend(neg_train_edge_labels)
        test_edge_embs.extend(neg_test_edge_embs)
        test_edge_labels.extend(neg_test_edge_labels)

        print(f'begin to evaluate run {i}!')
        
        auc, f1, precision, recall = clf_evaluate(train_edge_embs, train_edge_labels, test_edge_embs, test_edge_labels)
        
        print(f'finish evaluating run {i}!')
        
        # store these data too
        with open('record_with_twitter.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([i, auc, f1, precision, recall])

        print(f'auc: {auc}')
        print(f'f1: {f1}')
        print(f'precision: {precision}')
        print(f'recall: {recall}')

        aucs.append(auc)
        f1s.append(f1)
        precisions.append(precision)
        recalls.append(recall)

    with open('record_loading_with_twitter.dat', 'a') as f:
        f.write(f'auc: {np.mean(aucs)} +- {np.std(aucs)}\n')
        f.write(f'f1: {np.mean(f1s)} +- {np.std(f1s)}\n')
        f.write(f'precision: {np.mean(precisions)} +- {np.std(precisions)}\n')
        f.write(f'recall: {np.mean(recalls)} +- {np.std(recalls)}\n')
        f.write('\n')


if __name__ == "__main__":
    main()
