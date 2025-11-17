import torch
import numpy as np
import pandas as pd 
import scanpy as sc
import pickle 
import os

alpha  = 0.9
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epsilon = 0.0001


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def save_model(path, model):
    torch.save(model.state_dict(), path)


def load_embeddings(proj_name):
    """
    Load embeddings and gene expression data for a given project.

    Args:
        proj_name: Project/model name

    Returns:
        embedded_genes: Learned gene embeddings
        embedded_cells: Learned cell embeddings
        node_features: Original gene expression matrix
        out_features: Reconstructed gene expression matrix
        normalized_feature: Normalized gene expression matrix
    """
    model_dir = os.path.join("Models", proj_name)
    embedding_dir = os.path.join(model_dir, "Embeddings")
    
    embedded_genes = load_obj(os.path.join(embedding_dir, "gene_embedding"))
    embedded_cells = load_obj(os.path.join(embedding_dir, "cell_embedding"))
    out_features = load_obj(os.path.join(embedding_dir, "reconstructed_feature"))
    normalized_feature = load_obj(os.path.join(embedding_dir, "normalized_feature"))
    
    # Node features is optional (may not exist)
    node_features_path = os.path.join(model_dir, "Node_features", "node_features.pkl")
    if os.path.exists(node_features_path):
        node_features = pd.read_pickle(node_features_path)
    else:
        node_features = None
        print(f"Warning: node_features.pkl not found in {model_dir}/Node_features/")
    
    return embedded_genes, embedded_cells, node_features, out_features, normalized_feature