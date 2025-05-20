import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from viz import plot_confusion_matrix, plot_precision_recall_curve, plot_feature_importance
from viz import explain_gat, plot_attention_weights, plot_top_k_features

def evaluate_model(y, preds, probas, clf, labels, model_name="Logistic Regression"):
    print(f"Logistic Regression Accuracy: {accuracy_score(y, preds) * 100:.2f}%")
    print(f"Logistic Regression F1-Score: {f1_score(y, preds, average='macro') * 100:.2f}%")
    
    plot_confusion_matrix(y, preds, class_names=np.unique(labels))
    plot_precision_recall_curve(y, probas, class_names=np.unique(labels))

    importance = np.abs(clf.coef_).mean(axis=0)
    plot_feature_importance(importance, np.array([f'feat_{i}' for i in range(y.shape[0])]), model_name=model_name)

def evaluate_model(y, preds, probas, clf, labels, model_name="Model"):
    print(f"{model_name} Accuracy: {accuracy_score(y, preds) * 100:.2f}%")
    print(f"{model_name} F1-Score: {f1_score(y, preds, average='macro') * 100:.2f}%")
    
    plot_confusion_matrix(y, preds, class_names=np.unique(labels))
    plot_precision_recall_curve(y, probas, class_names=np.unique(labels))

    if hasattr(clf, 'coef_'):  # Logistic Regression feature importance
        importance = np.abs(clf.coef_).mean(axis=0)
    elif hasattr(clf, 'feature_importances_'):  # Random Forest feature importance
        importance = clf.feature_importances_
    else:
        importance = None

    if importance is not None:
        plot_feature_importance(importance, np.array([f'feat_{i}' for i in range(y.shape[0])]), model_name=model_name)

def evaluate_gcn(data, preds_gcn, probas_gcn, model, labels):
    print(f"GCN Accuracy: {accuracy_score(data.y.numpy(), preds_gcn.numpy()) * 100:.2f}%")
    print(f"GCN F1-Score: {f1_score(data.y.numpy(), preds_gcn.numpy(), average='macro') * 100:.2f}%")

    class_names = np.unique(labels)  # to ensure only unique classes
    plot_confusion_matrix(data.y.numpy(), preds_gcn.numpy(), class_names=class_names)
    plot_precision_recall_curve(data.y.numpy(), probas_gcn.numpy(), class_names=class_names)
    

def evaluate_gat(data, preds_gat, model, labels):
    print(f"GAT Accuracy: {accuracy_score(data.y.numpy(), preds_gat.numpy()) * 100:.2f}%")
    print(f"GAT F1-Score: {f1_score(data.y.numpy(), preds_gat.numpy(), average='macro') * 100:.2f}%")

    class_names = np.unique(labels)
    plot_confusion_matrix(data.y.numpy(), preds_gat.numpy(), class_names=class_names)

    explanation = explain_gat(model, data, node_idx=0)  # Explainability for a specific node
    plot_top_k_features(explanation, k=20)
    plot_attention_weights(model, data)