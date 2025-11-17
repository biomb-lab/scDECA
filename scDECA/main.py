import os
import pandas as pd
import numpy as np
import scanpy as sc
import torch
import pkg_resources
import random 
import warnings

from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges
from scDECA.data_loader import (
    construct_network, preprocess_data, create_knn_graph, 
    create_knn_loader, create_cell_loader, convert_nx_to_pyg_edges
)
from scDECA.train import train
from scDECA.utils import save_model

INTER_DIM = 512
EMBEDDING_DIM = 258
PROJECTION_DIM = 128
MAX_CELLS_BATCH_SIZE = 4000
MAX_CELLS_FOR_SPLITING = 10000
DE_GENES_NUM = 3000
NUM_LAYERS = 3
NUM_HEADS = 4

warnings.filterwarnings('ignore')


def run_scDECA(obj, model_type='scgpt', embedding_key=None, pre_processing_flag=True, 
              biogrid_flag=False, human_flag=False, number_of_batches=5, split_cells=False, 
              n_neighbors=25, max_epoch=150, model_name="", save_model_flag=False, 
              bbknn_flag=False, device_str="cuda:0", num_heads=8, 
              projection_dim=None):
    """
    Main function to run scDECA training pipeline.
    Loads gene embeddings from AnnData, constructs networks, and trains the model.
    Gene encoder uses addition fusion method.
    
    Args:
        obj: AnnData object containing single-cell data
        model_type: Type of foundation model ('scgpt', 'cellfm', 'mouse_geneformer', 'custom')
        embedding_key: Key for embeddings in adata.varm (auto-detected if None)
        pre_processing_flag: Whether to preprocess the data
        biogrid_flag: Whether to use BioGRID network (default: STRING)
        human_flag: Whether data is human (affects gene name casing)
        number_of_batches: Number of batches for training
        split_cells: Whether to split cells into batches (auto-detected for large datasets)
        n_neighbors: Number of neighbors for KNN graph
        max_epoch: Maximum number of training epochs
        model_name: Name for saving model outputs
        save_model_flag: Whether to save the trained model
        bbknn_flag: Whether to use BBKNN for batch correction
        device_str: Device string for training ('cuda:0', 'cpu', etc.)
        num_heads: Number of attention heads in cross-attention
        projection_dim: Projection dimension for feature fusion (auto-set if None)
    
    Returns:
        model: Trained scDECA model
        node_feature_fm: Foundation model gene embeddings DataFrame
        node_feature_raw: Raw gene expression DataFrame
        ppi_edge_index: PPI network edge indices
        
    Examples:
        # Using scGPT embeddings (assumes adata.varm['scGPT_gene_token'] exists)
        model, fm_emb, raw_exp, ppi = run_scDECA(adata, model_type='scgpt')

        # Using custom embeddings with specific key
        model, fm_emb, raw_exp, ppi = run_scDECA(
            adata, model_type='custom', embedding_key='my_gene_embeddings'
        )
        
        # With custom hyperparameters
        model, fm_emb, raw_exp, ppi = run_scDECA(
            adata, 
            model_type='scgpt',
            num_heads=16,
            max_epoch=300
        )
    """
    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    global NUM_HEADS
    NUM_HEADS = num_heads
    
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Device set to: {device}")
    print(f"Using {model_type.upper()} embeddings from AnnData.varm")
    
    default_embedding_keys = {
        'scgpt': 'scGPT_gene_token',
        'cellfm': 'CellFM_gene_token', 
        'custom': 'custom_gene_embedding'
    }
    
    if embedding_key is None:
        if model_type.lower() in default_embedding_keys:
            embedding_key = default_embedding_keys[model_type.lower()]
        else:
            raise ValueError(f"No default embedding key for model type '{model_type}'. Please specify embedding_key.")
    
    if embedding_key not in obj.varm:
        available_keys = list(obj.varm.keys())
        raise KeyError(f"Embedding key '{embedding_key}' not found in adata.varm. Available keys: {available_keys}")
    
    print(f"Using embedding key: '{embedding_key}'")
    print(f"Embedding shape: {obj.varm[embedding_key].shape}")
    
    # Create model-specific directories
    package_dir = os.path.dirname(__file__)             
    models_root = os.path.join(package_dir, "Models")   
    model_dir = os.path.join(models_root, model_name)   

    node_features_dir = os.path.join(model_dir, "Node_features")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(node_features_dir, exist_ok=True)
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(node_features_dir, exist_ok=True)
    
    
    if pre_processing_flag:
        obj = preprocess_data(obj, n_neighbors)
    else:
        if obj.raw is None:
            obj.raw = obj.copy()
            sc.pp.log1p(obj)
        else:
            print("obj.raw already exists; skipping log1p")
        obj.X = obj.raw.X
        if not bbknn_flag:
            sc.pp.neighbors(obj, n_neighbors=n_neighbors, n_pcs=15)
    
    if obj.obs.shape[0] > MAX_CELLS_FOR_SPLITING:
        split_cells = True
    
    if split_cells:
        batch_size = obj.obs.shape[0] // number_of_batches
        if batch_size > MAX_CELLS_BATCH_SIZE:
            number_of_batches = obj.obs.shape[0] // MAX_CELLS_BATCH_SIZE

    if not biogrid_flag:
        print("Loading PPI network...")
        net = pd.read_csv(pkg_resources.resource_filename(__name__, r"Data/format_h_sapiens.csv"))[["g1_symbol", "g2_symbol", "conn"]].drop_duplicates()
        net, ppi, node_feature_fm, node_feature_raw = construct_network(
            obj, net, model_type, adata=obj, embedding_key=embedding_key, human_flag=human_flag
        )
        print(f"Network built with {model_type.upper()} embeddings from AnnData:")
        print(f"   - FM embeddings: {node_feature_fm.shape}")
        print(f"   - Raw expression: {node_feature_raw.shape}")
    else:
        print("Loading BioGRID network...")
        net = pd.read_table(pkg_resources.resource_filename(__name__, r"Data/BIOGRID.tab.txt"))[["OFFICIAL_SYMBOL_A", "OFFICIAL_SYMBOL_B"]].drop_duplicates()
        net, ppi, node_feature_fm, node_feature_raw = construct_network(
            obj, net, model_type, adata=obj, embedding_key=embedding_key, biogrid_flag=biogrid_flag, human_flag=human_flag
        )
        print(f"Network built with {model_type.upper()} embeddings from AnnData:")
        print(f"   - FM embeddings: {node_feature_fm.shape}")
        print(f"   - Raw expression: {node_feature_raw.shape}")
    
    ppi_edge_index, _ = convert_nx_to_pyg_edges(ppi)
    ppi_edge_index = ppi_edge_index.to(device)

    if split_cells:
        obj = obj[:, node_feature_fm.index]
        sc.pp.highly_variable_genes(obj, n_top_genes=DE_GENES_NUM)
        highly_variable_index = obj.var.highly_variable 
        if highly_variable_index.sum() < 1000 or highly_variable_index.sum() > 5000:
            obj.var["std"] = sc.get.obs_df(obj.raw.to_adata(), list(obj.var.index)).std()
            highly_variable_index = obj.var["std"] >= obj.var["std"].sort_values(ascending=False)[3500]
        
        print(f"Highly variable genes: {highly_variable_index.sum()}")
    else:
        obj = obj[:, node_feature_fm.index]
        knn_edge_index, highly_variable_index = create_knn_graph(obj)    
        loader = create_knn_loader(knn_edge_index, knn_edge_index.shape[1] // number_of_batches)
  
    highly_variable_index = highly_variable_index[node_feature_fm.index]
    
    # Save node features to model-specific directory
    node_feature_fm.to_pickle(os.path.join(node_features_dir, "node_features_fm.pkl"))
    node_feature_raw.to_pickle(os.path.join(node_features_dir, "node_features.pkl"))
    
    hvg_index_path = os.path.join(node_features_dir, "hvg_index.pkl")
    hvg_mask = highly_variable_index.loc[highly_variable_index.index.isin(node_feature_fm.index)]
    hvg_mask = hvg_mask.loc[node_feature_fm.index]
    hvg_mask.to_pickle(hvg_index_path)
    
    print(f"Saved HVG index to: {hvg_index_path}")
    print(f"Saved FM embeddings to: {os.path.join(node_features_dir, 'node_features_fm.pkl')}")
    print(f"Saved raw expression to: {os.path.join(node_features_dir, 'node_features.pkl')}")

    print("Preparing dual input data...")
    
    x_foundation = node_feature_fm.values
    x_foundation = torch.tensor(x_foundation, dtype=torch.float32).cpu()
    
    x_raw = node_feature_raw.values
    x_raw = torch.tensor(x_raw, dtype=torch.float32).cpu()
    
    print(f"Dual input data prepared:")
    print(f"   - {model_type.upper()} embeddings shape: {x_foundation.shape}")
    print(f"   - Raw expression shape: {x_raw.shape}")
    print(f"   - Genes: {x_foundation.shape[0]}")
    print(f"   - Cells: {x_raw.shape[1]}")
    print(f"   - {model_type.upper()} embedding dimension: {x_foundation.shape[1]}")
    
    if split_cells: 
        loader = create_cell_loader(x_raw, obj.obsp["distances"], x_raw.shape[1] // number_of_batches)

    print("Preparing gene and cell names for attention analysis...")
    
    gene_names = node_feature_fm.index.tolist()
    
    if hasattr(obj.obs, 'index'):
        cell_names = obj.obs.index.tolist()
    else:
        cell_names = [f"Cell_{i}" for i in range(obj.n_obs)]
    
    print(f"Prepared {len(gene_names)} gene names and {len(cell_names)} cell names")

    data = Data(x_foundation, ppi_edge_index)
    data = train_test_split_edges(data, test_ratio=0.2, val_ratio=0)
    
    print(f"Starting training:")
    print(f"   - Data.x ({model_type.upper()}): {data.x.shape}")
    print(f"   - x_raw: {x_raw.shape}")
    print(f"   - Split cells: {split_cells}")
    print(f"   - Number of batches: {number_of_batches}")
    
    model = train(
        data, loader, highly_variable_index, x_raw, device,
        model_name, number_of_batches=number_of_batches, 
        max_epoch=max_epoch, network_reduction_interval=30, 
        cell_batching_flag=split_cells, enable_attention_analysis=True,
        projection_dim=projection_dim, cell_names=cell_names, gene_names=gene_names,
        inter_dim=INTER_DIM, embedding_dim=EMBEDDING_DIM, 
        num_layers=NUM_LAYERS, num_heads=NUM_HEADS
    )
    
    if save_model_flag:
        save_model(os.path.join(model_dir, "model.pt"), model)

    return model