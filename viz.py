import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
import networkx as nx
import torch

from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve, average_precision_score
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.config import ModelConfig, ExplanationType, MaskType

def plot_confusion_matrix(y_true, y_pred, class_names=None, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title(title)
    plt.grid(False)
    plt.tight_layout()
    plt.show()

def plot_precision_recall_curve(y_true, y_scores, class_names):
    # Binarize labels (for one-vs-rest)
    y_bin = label_binarize(y_true, classes=np.arange(len(class_names)))
    plt.figure(figsize=(10, 6))

    for i in range(len(class_names)):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_scores[:, i])
        ap = average_precision_score(y_bin[:, i], y_scores[:, i])
        plt.plot(recall, precision, label=f"{class_names[i]} (AP={ap:.2f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Per Class)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_feature_importance(importance, feature_names, model_name="Model", top_n=10):
    top_idx = np.argsort(importance)[-top_n:]
    colors = plt.cm.viridis(np.linspace(0, 1, top_n))  # Using a colormap to assign unique colors

    plt.figure(figsize=(8, 5))
    plt.barh(range(top_n), importance[top_idx], color=colors)  # Apply colors
    plt.yticks(range(top_n), feature_names[top_idx])
    plt.xlabel("Feature Importance")
    plt.title(f"{model_name} Feature Importance (Top {top_n})")
    plt.tight_layout()
    plt.show()


def plot_gcn_embeddings_tsne(embeddings, labels, title="GCN Embeddings (t-SNE)", perplexity=30):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    emb_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=emb_2d[:, 0], y=emb_2d[:, 1], hue=labels, palette='tab10', legend='full')
    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()
    plt.show()

def plot_top_k_features(explanation, k=20):
    feature_mask = explanation.node_mask.cpu().detach().numpy()

    # Flatten if multidimensional
    if feature_mask.ndim > 1:
        feature_mask = feature_mask.mean(axis=0)

    assert feature_mask.ndim == 1, f"Expected 1D feature importance, got shape {feature_mask.shape}"

    # Select top-k features
    top_k_indices = np.argsort(feature_mask)[-k:][::-1]
    top_k_scores = feature_mask[top_k_indices]

    # Generate distinct colors using a colormap
    colors = plt.cm.viridis(np.linspace(0, 1, k))  # You can try 'plasma', 'tab10', or 'coolwarm'

    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(range(k), top_k_scores, align='center', color=colors)  # Apply colormap for distinction
    plt.yticks(range(k), [f"Feature {i}" for i in top_k_indices])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {k} Most Important Features')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

def explain_gcn(data, model, node_idx=0):
    model_config = ModelConfig(
        mode='multiclass_classification',
        task_level='node',
        return_type='log_probs'
    )

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=100),
        explanation_type=ExplanationType.model,
        model_config=model_config,
        node_mask_type=MaskType.attributes,
        edge_mask_type=MaskType.object
    )

    explanation = explainer(x=data.x, edge_index=data.edge_index, index=node_idx)
    plot_top_k_features(explanation, k=20)



def plot_attention_weights(model, data):
    """Visualize attention weights from a trained GAT model."""
    model.eval()
    with torch.no_grad():
        # Call the layer with return_attention_weights=True
        _, (edge_index, attn_weights) = model.gat1(data.x, data.edge_index, return_attention_weights=True)

    attn_weights = attn_weights.cpu().numpy()
    plt.figure(figsize=(8, 6))
    plt.hist(attn_weights.flatten(), bins=50, alpha=0.75, color="blue")
    plt.xlabel("Attention Weight")
    plt.ylabel("Frequency")
    plt.title("Distribution of Attention Weights (GAT Layer 1)")
    plt.tight_layout()
    plt.show()

def explain_gat(model, data, node_idx=0):
    """Explain model predictions using GNNExplainer."""
    model_config = ModelConfig(
        mode='multiclass_classification',
        task_level='node',
        return_type='log_probs'
    )

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=100),
        explanation_type=ExplanationType.model,
        model_config=model_config,
        node_mask_type=MaskType.attributes,
        edge_mask_type=MaskType.object
    )

    explanation = explainer(x=data.x, edge_index=data.edge_index, index=node_idx)
    return explanation

def plot_class_distribution(labels, dataset_name="Dataset"):
    """Visualizes class distribution with distinct colors."""
    label_df = pd.DataFrame(labels, columns=["class_label"])
    
    # Count occurrences of each class
    label_counts = label_df["class_label"].value_counts()
    
    # Generate a color map with unique colors per class
    colors = cm.get_cmap('tab10', len(label_counts))  # Adjusted to match label count
    
    # Plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(label_counts.index, label_counts.values, color=[colors(i) for i in range(len(label_counts))])
    
    plt.title(f"Class Distribution in {dataset_name}")
    plt.xlabel("Class Label")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_citation_graph(edges, num_edges=50, dataset_name="Cora Citation Graph"):
    """Visualizes a citation graph using NetworkX."""
    G = nx.DiGraph()
    G.add_edges_from(edges[:num_edges])  # Limit edges for clarity

    # Draw the graph
    plt.figure(figsize=(10, 7))
    pos = nx.spring_layout(G, seed=42)  # Consistent layout for reproducibility
    nx.draw(G, pos, node_size=50, arrowsize=10, with_labels=False)

    # Rotate and adjust labels manually to prevent overlap
    label_offset = 0.05
    for label, (x, y) in pos.items():
        plt.text(x + label_offset, y + label_offset, str(label), fontsize=8, ha='center', va='center', rotation=45)

    plt.title(f"{dataset_name} (first {num_edges} edges)")
    plt.savefig("citation_graph.png")
    plt.show()

def plot_citation_subgraph(edges, labels, subset_size=100, dataset_name="Cora Citation Graph"):
    """Visualizes a subgraph of citations with nodes colored by class labels."""
    subset_indices = np.arange(subset_size)

    # Create a subgraph based on paper indices in the subset
    subset_edges = [tuple(edge) for edge in edges if edge[0] in subset_indices and edge[1] in subset_indices]

    G = nx.DiGraph()
    G.add_edges_from(subset_edges)

    # Map node to label name (original class label)
    index_to_label = {i: label for i, label in enumerate(labels)}
    node_labels = {i: index_to_label[i] for i in G.nodes}

    # Assign a color to each label class
    unique_labels = list(set(index_to_label.values()))
    label_to_color = {label: idx for idx, label in enumerate(unique_labels)}
    node_colors = [label_to_color[index_to_label[i]] for i in G.nodes]

    # Visualization
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, seed=42)

    # Draw nodes with color by class
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors, cmap=plt.cm.tab10)

    # Draw edges
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=12, edge_color='gray')

    # Draw node labels (use class names like 'Neural_Networks')
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)

    # Legend
    handles = [mpatches.Patch(color=plt.cm.tab10(label_to_color[l] / len(unique_labels)), label=l) for l in unique_labels]
    plt.legend(handles=handles, title="Class Labels", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title(f"{dataset_name} (first {subset_size} nodes with class labels)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()