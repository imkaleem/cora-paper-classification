import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from graph_models import GAT, GCN
import torch
import torch.nn.functional as F

def train_logistic_regression(X, y, skf, max_iterations=1000):
    preds_lr = np.zeros_like(y)
    probas_lr = np.zeros((len(y), len(np.unique(y))))

    for train_idx, test_idx in skf:
        clf = LogisticRegression(max_iter=max_iterations)
        clf.fit(X[train_idx], y[train_idx])
        preds_lr[test_idx] = clf.predict(X[test_idx])
        probas_lr[test_idx] = clf.predict_proba(X[test_idx])

    acc = accuracy_score(y, preds_lr)
    f1 = f1_score(y, preds_lr, average="macro")

    return preds_lr, probas_lr, acc, f1, clf

def train_random_forest(X, y, skf, n_estimators=100, random_state=42):
    preds_rf = np.zeros_like(y)
    probas_rf = np.zeros((len(y), len(np.unique(y))))

    for train_idx, test_idx in skf:
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        clf.fit(X[train_idx], y[train_idx])
        preds_rf[test_idx] = clf.predict(X[test_idx])
        probas_rf[test_idx] = clf.predict_proba(X[test_idx])

    acc = accuracy_score(y, preds_rf)
    f1 = f1_score(y, preds_rf, average="macro")

    return preds_rf, probas_rf, acc, f1, clf

def train_gcn(data, masks, num_epochs=200):
    num_classes = len(torch.unique(data.y))
    preds_gcn = torch.zeros(data.num_nodes, dtype=torch.int)
    probas_gcn = torch.zeros((data.num_nodes, num_classes))

    for fold, (train_mask, test_mask) in enumerate(masks):
        print(f"Fold: {fold}")

        model = GCN(
            in_channels=data.num_node_features,
            hidden_channels=16,
            out_channels=num_classes
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        # Training Loop
        model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.nll_loss(out[train_mask], data.y[train_mask])
            loss.backward()
            optimizer.step()

        # Evaluation
        model.eval()
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        #print(preds_gcn.dtype, pred.dtype)
        #preds_gcn[test_mask] = pred[test_mask].cpu()
        preds_gcn[test_mask] = pred[test_mask].cpu().to(preds_gcn.dtype)
        probas_gcn[test_mask] = out[test_mask].exp().detach().cpu()

    # Metrics
    acc = accuracy_score(data.y.numpy(), preds_gcn.numpy())
    f1 = f1_score(data.y.numpy(), preds_gcn.numpy(), average="macro")
    
    return preds_gcn, probas_gcn, acc, f1, model


def train_gat(data, masks, num_epochs=50):
    preds_gat = torch.zeros(data.num_nodes, dtype=torch.long)

    for fold, (train_mask, test_mask) in enumerate(masks):
        print(f"Fold: {fold}")

        model = GAT(in_channels=data.num_node_features, hidden_channels=8, out_channels=len(torch.unique(data.y)))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

        # Training Loop
        model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.nll_loss(out[train_mask], data.y[train_mask])
            loss.backward()
            optimizer.step()

        # Evaluation
        model.eval()
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        preds_gat[test_mask] = pred[test_mask].cpu()

    acc = accuracy_score(data.y.numpy(), preds_gat.numpy())
    f1 = f1_score(data.y.numpy(), preds_gat.numpy(), average="macro")
    
    return preds_gat, acc, f1, model
