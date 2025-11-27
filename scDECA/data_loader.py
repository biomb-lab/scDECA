import torch
import pandas as pd
import numpy as np
import scanpy as sc
import networkx as nx
from torch_geometric.utils import convert
from torch_geometric.data import Data
from torch.utils.data import DataLoader, Dataset



def extract_embeddings_from_anndata(adata, embedding_key, target_genes=None, human_flag=True):
    """
    Extract gene embeddings from AnnData object's varm attribute.
    Filters out NaN values and handles mouse/human gene name casing.
    
    Args:
        adata: AnnData object containing embeddings in .varm
        embedding_key: Key for embeddings in adata.varm
        target_genes: Optional list of target genes to filter
        human_flag: If False, performs mouse gene casing conversion
        
    Returns:
        gene_embedding_dict: Dictionary mapping genes to embeddings
        embedding_df: DataFrame with genes as index
        available_genes: List of genes with valid embeddings
        missing_genes: List of genes without valid embeddings
    """
    print(f"Loading {embedding_key} gene embeddings from AnnData...")
    
    if embedding_key not in adata.varm:
        available_keys = list(adata.varm.keys())
        raise KeyError(f"Embedding key '{embedding_key}' not found in adata.varm. Available keys: {available_keys}")
    
    gene_embeddings = adata.varm[embedding_key]
    all_genes = adata.var.index.tolist()
    
    print(f"Found embeddings in adata.varm['{embedding_key}']")
    print(f"   - Shape: {gene_embeddings.shape}")
    print(f"   - Total genes: {len(all_genes)}")
    
    nan_mask = np.isnan(gene_embeddings).any(axis=1)
    valid_gene_indices = ~nan_mask
    
    print(f"   - Genes with valid embeddings (no NaN): {valid_gene_indices.sum()}")
    print(f"   - Genes with NaN embeddings: {nan_mask.sum()}")
    
    valid_genes = [all_genes[i] for i in range(len(all_genes)) if valid_gene_indices[i]]
    valid_embeddings = gene_embeddings[valid_gene_indices]
    
    if target_genes is not None:
        if not human_flag:
            valid_genes_upper = [gene.upper() for gene in valid_genes]
            available_genes_upper = [gene for gene in target_genes if gene in valid_genes_upper]
            missing_genes = [gene for gene in target_genes if gene not in valid_genes_upper]
            
            upper_to_original = {gene.upper(): gene for gene in valid_genes}
            available_genes = [upper_to_original[gene] for gene in available_genes_upper]
            
            print(f"Mouse casing conversion:")
            print(f"   - target_genes (upper): {target_genes[:5] if len(target_genes) > 5 else target_genes}")
            print(f"   - valid_genes (mouse): {valid_genes[:5]}")
            print(f"   - available_genes (mouse): {available_genes[:5]}")
        else:
            available_genes = [gene for gene in target_genes if gene in valid_genes]
            missing_genes = [gene for gene in target_genes if gene not in valid_genes]
        
        print(f"Found {len(available_genes)} target genes with valid embeddings")
        if missing_genes:
            print(f"WARNING: {len(missing_genes)} target genes missing or have NaN embeddings")
            if len(missing_genes) <= 10:
                print(f"   Missing/NaN genes: {missing_genes}")
            else:
                print(f"   First 10 missing/NaN genes: {missing_genes[:10]}...")
        
        gene_indices = [valid_genes.index(gene) for gene in available_genes]
        filtered_embeddings = valid_embeddings[gene_indices]
        
    else:
        available_genes = valid_genes
        missing_genes = []
        filtered_embeddings = valid_embeddings
    
    if len(available_genes) == 0:
        raise ValueError("No genes available after filtering NaN values!")
    
    gene_embedding_dict = {gene: filtered_embeddings[i] for i, gene in enumerate(available_genes)}
    
    embedding_df = pd.DataFrame(
        filtered_embeddings, 
        index=available_genes,
        columns=[f"{embedding_key}_dim_{i}" for i in range(filtered_embeddings.shape[1])]
    )
    
    print(f"Created gene embeddings from AnnData: {embedding_df.shape}")
    print(f"   - Final genes after NaN filtering: {len(available_genes)}")
    print(f"   - Gene index sample: {embedding_df.index[:5].tolist()}")
    
    return gene_embedding_dict, embedding_df, available_genes, missing_genes


def prepare_gene_embeddings(model_type, target_genes, adata=None, embedding_key=None, human_flag=True):
    print(f"Preparing {model_type.upper()} gene embeddings for {len(target_genes)} target genes...")
    
    if adata is None:
        raise ValueError("AnnData object is required")
    
    default_embedding_keys = {
        'scgpt': 'scGPT_gene_token',
        'cellfm': 'CellFM_gene_token',
        'mouse_geneformer': 'mouse_geneformer',
        'custom': 'custom_gene_embedding'
    }
    
    if embedding_key is None:
        if model_type.lower() in default_embedding_keys:
            embedding_key = default_embedding_keys[model_type.lower()]
        else:
            raise ValueError(f"No default embedding key for model type '{model_type}'. Please specify embedding_key.")
    
    return extract_embeddings_from_anndata(adata, embedding_key, target_genes, human_flag)


def construct_network(obj, net, model_type, adata=None, embedding_key=None, biogrid_flag=False, human_flag=False, 
                      expression_cutoff=None, network_cutoff=None):
    """
    Construct gene-gene network with foundation model embeddings and raw expression.
    Integrates PPI network with single-cell data and gene embeddings.
    
    Args:
        obj: AnnData object with single-cell data
        net: DataFrame containing PPI network edges
        model_type: Type of foundation model ('scgpt', 'cellfm', etc.)
        adata: AnnData object containing gene embeddings (default: same as obj)
        embedding_key: Key for embeddings in adata.varm
        biogrid_flag: Whether network is in BioGRID format
        human_flag: If False, applies mouse gene casing conversion
        
    Returns:
        net: Filtered network DataFrame
        gp: NetworkX graph object
        node_feature_fm: Foundation model gene embeddings DataFrame
        node_feature_raw: Raw gene expression DataFrame
    """
    print(f"Constructing network with {model_type.upper()} embeddings from AnnData...")

    if adata is None:
        adata = obj

    if not biogrid_flag:
        net.columns = ["Source", "Target", "Conn"]
        net = net.loc[net.Conn >= network_cutoff] 
    else:
        net.columns = ["Source", "Target"]

    if not human_flag:
        print("Before casing:")
        print(net["Source"].unique()[:5])
        net["Source"] = net["Source"].apply(lambda x: x[0] + x[1:].lower()).astype(str)
        net["Target"] = net["Target"].apply(lambda x: x[0] + x[1:].lower()).astype(str)
        print("After casing:")
        print(net["Source"].unique()[:5])

    all_net_genes = list(pd.concat([net.Source, net.Target]).drop_duplicates())
    sc_genes = obj.var[obj.var.index.isin(all_net_genes)].index.tolist()

    print(f"Gene filtering process:")
    print(f"   - PPI network genes: {len(all_net_genes)}")
    print(f"   - Genes in single-cell data: {len(sc_genes)}")

    print("Applying expression cutoff...")
    node_feature_raw = sc.get.obs_df(obj.raw.to_adata(), sc_genes).T
    node_feature_raw["non_zero"] = node_feature_raw.astype(bool).sum(axis=1)
    node_feature_filtered = node_feature_raw[
        node_feature_raw["non_zero"] > node_feature_raw.shape[1] * expression_cutoff
    ].drop("non_zero", axis=1)

    selected_genes = node_feature_filtered.index.tolist()
    print(f"   - Genes after expression cutoff: {len(selected_genes)}")

    net = net.loc[(net.Source != net.Target)]
    net = net.loc[net.Source.isin(selected_genes) & net.Target.isin(selected_genes)]
    gp = nx.from_pandas_edgelist(net, "Source", "Target")

    if not human_flag:
        selected_genes_upper = [g.upper() for g in selected_genes]
    else:
        selected_genes_upper = selected_genes
    print(f"Number of genes passed to embedding: {len(selected_genes_upper)}")

    _, embedding_df, available_genes, missing_genes = prepare_gene_embeddings(
        model_type, selected_genes_upper, adata=adata, embedding_key=embedding_key, human_flag=human_flag
    )

    print("selected_genes:", selected_genes[:10])
    print("selected_genes_upper:", selected_genes_upper[:10])
    print("embedding_df.index (final):", embedding_df.index[:10])
    print("missing genes:", missing_genes)
    print(f"Number of genes returned from embedding: {embedding_df.shape[0]}")
    print(f"Missing genes: {len(missing_genes)}")

    net = net.loc[net.Source.isin(embedding_df.index) & net.Target.isin(embedding_df.index)]
    gp = nx.from_pandas_edgelist(net, "Source", "Target")
    final_genes = list(gp.nodes)

    node_feature_fm = embedding_df.loc[final_genes]
    node_feature_raw_filtered = node_feature_filtered.loc[final_genes]
    
    print(f"Network built successfully:")
    print(f"   - Network nodes: {len(gp.nodes)}")
    print(f"   - Network edges: {len(gp.edges)}")
    print(f"   - Foundation model feature shape: {node_feature_fm.shape}")
    print(f"   - Raw expression feature shape: {node_feature_raw_filtered.shape}")
    print(f"   - {model_type.upper()} embedding dimension: {node_feature_fm.shape[1]}")
    print(f"   - Number of cells: {node_feature_raw_filtered.shape[1]}")

    return net, gp, node_feature_fm, node_feature_raw_filtered


def preprocess_data(adata, n_neighbors): 
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
   
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata.copy()
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=15)

    return adata


def generate_knn_batch(knn, idxs, k=15):
    idxs = idxs.cpu().numpy()
    adjacency_matrix = torch.tensor(knn[idxs][:, idxs].toarray())
    row_indices, col_indices = torch.nonzero(adjacency_matrix, as_tuple=True)
    knn_edge_index = torch.stack((row_indices, col_indices))
    knn_edge_index = torch.unique(knn_edge_index, dim=1)
    return knn_edge_index


def create_knn_graph(obj, de_genes_num=3000):
    graph = obj.obsp["distances"].toarray()
    graph = (graph > 0).astype(int)
    graph = nx.from_numpy_array(np.matrix(graph))
    ppi_geo = convert.from_networkx(graph)
    edge_index = ppi_geo.edge_index
    sc.pp.highly_variable_genes(obj, n_top_genes=de_genes_num)
    return edge_index, obj.var.highly_variable


def create_knn_loader(edge_index, batch_size):
    knn_dataset = KNNDataset(edge_index)
    knn_loader = DataLoader(knn_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    return knn_loader


def create_cell_loader(x, edge_index, batch_size):
    cell_dataset = CellDataset(x, edge_index)
    cell_loader = DataLoader(cell_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    return cell_loader


def convert_nx_to_pyg_edges(G, mapping=None):
    G = G.to_directed() if not nx.is_directed(G) else G
    if mapping is None:  
        mapping = dict(zip(G.nodes(), range(G.number_of_nodes())))
    edge_index = torch.empty((2, G.number_of_edges()), dtype=torch.long)
    for i, (src, dst) in enumerate(G.edges()):
        edge_index[0, i] = mapping[src]
        edge_index[1, i] = mapping[dst]
    return edge_index, mapping


class KNNDataset(Dataset):
    def __init__(self, edge_index):
        self.edge_index = edge_index.T

    def __len__(self):
        return self.edge_index.shape[0]

    def __getitem__(self, idx):
        return self.edge_index[idx,:]

            
class CellDataset(Dataset):
    def __init__(self, x, knn):
        self.x = x
        self.knn = knn


    def __len__(self):
        return self.x.shape[1]

    def __getitem__(self, idx):
        return self.x[:,idx] , idx

    
class CustomDataset(Dataset):
    def __init__(self, x):
        self.data = x
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return torch.tensor(self.data[index])