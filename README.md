# **Graph-Based Machine Learning for Node Classification**

## Overview
This project explores **node classification** using a range of machine learning techniques, including:
- **Logistic Regression**
- **Random Forest**
- **Graph Convolutional Networks (GCN)**
- **Graph Attention Networks (GAT)**

All function calls—from **data preprocessing** to **exploratory data analysis (EDA)**, **feature engineering**, and **model training**—are centralized within `task_clean_new.ipynb`.

---

## Project Structure

| File                    | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| `data_preparation.py`   | Handles data preprocessing steps such as feature extraction and dataset cleaning. |
| `evaluate.py`           | Implements model evaluation metrics, including accuracy and F1-score calculations. |
| `graph_models.py`       | Contains implementations of graph-based models like GCN and GAT.           |
| `predictions.tsv`       | Stores general model predictions for later analysis.                        |
| `predictions_gat.tsv`   | Predictions generated using the GAT model.                                  |
| `predictions_gcn.tsv`   | Predictions generated using the GCN model.                                  |
| `predictions_gcns.tsv`  | Extended GCN model predictions.                        |
| `predictions_lr.tsv`    | Predictions from Logistic Regression.                                       |
| `predictions_rf.tsv`    | Predictions from Random Forest.                                             |
| `requirements.txt`      | Lists necessary dependencies to run the project.                            |
| `task_clean_new.ipynb`  | **Main notebook**: integrates all function calls for data processing, EDA, feature preparation, and model training. |
| `train.py`              | Script for training machine learning models, including logistic regression, random forests, and GNN models. |
| `viz.py`                | Contains visualization functions for EDA, graph structures, and feature importance plots. |

---

## How to Use

### **1. Setup Environment**
Install required dependencies:
```
pip install -r requirements.txt
```

### **2. Run Jupyter Notebook**
Launch and run the main notebook:

```
jupyter notebook task_clean_new.ipynb
```
### **3. Explore Model Predictions**
After running the models, check the prediction outputs in the `.tsv` files for evaluation and submission.

---

## Model Implementation Breakdown

### **1. Data Preparation**
Handled by `data_preparation.py`, includes:
- Feature extraction
- Graph construction  

### **2. Model Training**
- Logistic Regression and Random Forest (via `train.py`)  
- GCN and GAT models (via `graph_models.py`)  
- Unified in `task_clean_new.ipynb`

### **3. Evaluation & Visualization**
- Accuracy and F1 Score via `evaluate.py`  
- Visual insights with `viz.py`

---

## Future Improvements
- Optimize hyperparameters for better GCN/GAT performance  
- Explore additional feature engineering techniques  
- Scale models for larger citation networks
