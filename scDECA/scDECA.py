import torch
import torch.nn as nn
from torch_geometric.nn import InnerProductDecoder
from torch_geometric.utils import negative_sampling
from sklearn.metrics import average_precision_score, roc_auc_score

from scDECA.model import (
    ExpressionDecoder, GeneEncoder, CellEncoder, 
    CrossAttentionEncoder, StructuralEncoder
)

EPS = 1e-15


class scDECA(torch.nn.Module):
    """
    scDECA: single-cell Dual-Encoder with Cross-Attention
    
    Main fusion model integrating foundation model embeddings with graph-based approaches
    for single-cell RNA-seq analysis. Combines:
    - Gene encoder: Fuses foundation embeddings with expression via PPI network (add fusion)
    - Cell encoder: Processes cell-cell relationships via KNN network  
    - Cross-attention: Enables bidirectional gene-cell information flow
    - Structural encoders: Final task-specific encoding for genes and cells
    
    Uses optimized activation functions for each component:
    - ExpressionDecoder: GELU for smooth reconstruction
    - Graph layers: ELU for stable gradient flow
    - Attention FFN: Swish for selective gating
    - Structural encoder: Mish for high performance
    
    Args:
        num_cells: Number of cells
        num_genes: Number of genes
        scgpt_embed_dim: Foundation model embedding dimension
        inter_gene_dim: Intermediate dimension for gene encoder
        embd_gene_dim: Output embedding dimension for genes
        inter_cell_dim: Intermediate dimension for cell encoder
        embd_cell_dim: Output embedding dimension for cells
        lambda_genes: Weight for gene reconstruction loss
        lambda_cells: Weight for cell reconstruction loss
        num_layers: Number of graph convolution layers
        drop_p: Dropout probability
        num_heads: Number of attention heads
        projection_dim: Projection dimension for fusion
    """
    def __init__(self, num_cells, num_genes, scgpt_embed_dim, inter_gene_dim, embd_gene_dim, 
                 inter_cell_dim, embd_cell_dim, lambda_genes=1, lambda_cells=1, 
                 num_layers=2, drop_p=0.25, num_heads=8, projection_dim=None):
        super(scDECA, self).__init__()
        
        self.num_cells = num_cells
        self.num_genes = num_genes
        self.scgpt_embed_dim = scgpt_embed_dim
        self.inter_gene_dim = inter_gene_dim
        self.embd_gene_dim = embd_gene_dim
        self.inter_cell_dim = inter_cell_dim
        self.embd_cell_dim = embd_cell_dim
        self.lambda_genes = lambda_genes
        self.lambda_cells = lambda_cells

        self.shared_embed_dim = max(embd_gene_dim, embd_cell_dim)
        
        if projection_dim is None:
            projection_dim = self.shared_embed_dim * 3
            print(f"Auto-set projection_dim to: {projection_dim}")
        
        # Gene encoder with foundation model fusion (add fusion only)
        self.gene_encoder = GeneEncoder(
            scgpt_embed_dim, 
            num_cells,  # expression_dim
            self.shared_embed_dim, 
            num_layers, 
            drop_p,
            projection_dim=projection_dim
        )

        # Cell encoder with KNN graph
        self.cell_encoder = CellEncoder(num_genes, self.shared_embed_dim, num_layers, drop_p)
        
        # Cross-attention for gene-cell interaction
        self.cross_attention_encoder = CrossAttentionEncoder(
            self.shared_embed_dim, self.shared_embed_dim, self.shared_embed_dim, 
            num_heads, num_layers, drop_p
        )
        
        # Final structural encoders
        self.gene_final_encoder = StructuralEncoder(self.shared_embed_dim, inter_gene_dim, embd_gene_dim, drop_p=drop_p, scale_param=None, reducer=False)
        self.cell_final_encoder = StructuralEncoder(self.shared_embed_dim, inter_cell_dim, embd_cell_dim, drop_p=drop_p, reducer=True)
        
        # Expression decoder and loss
        self.feature_decoder = ExpressionDecoder(num_genes, embd_cell_dim, inter_cell_dim, drop_p=0)
        self.ipd = InnerProductDecoder()
        self.feature_criterion = nn.MSELoss(reduction='mean')

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None, sig=False):
        """
        Calculate reconstruction loss for graph edges.
        
        Args:
            z: Node embeddings
            pos_edge_index: Positive edges
            neg_edge_index: Negative edges (sampled if None)
            sig: Whether to use sigmoid activation
            
        Returns:
            Combined positive and negative edge reconstruction loss
        """
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))

        if not sig:
            embd = torch.corrcoef(z)
            pos = torch.sigmoid(embd[pos_edge_index[0], pos_edge_index[1]])
            neg = torch.sigmoid(embd[neg_edge_index[0], neg_edge_index[1]])
            pos_loss = -torch.log(pos + EPS).mean()
            neg_loss = -torch.log(1 - neg + EPS).mean()
        else:
            pos_loss = -torch.log(self.ipd(z, pos_edge_index, sigmoid=sig) + EPS).mean()
            neg_loss = -torch.log(1 - self.ipd(z, neg_edge_index, sigmoid=sig) + EPS).mean()

        return pos_loss + neg_loss

    def kl_loss(self, mu=None, logstd=None):
        """
        Calculate KL divergence loss (for variational models).
        Currently unused but kept for compatibility.
        """
        mu = self.gene_final_encoder.__mu__ if mu is None else mu
        logstd = self.gene_final_encoder.__logstd__ if logstd is None else logstd
        return -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))
  
    def test(self, z, pos_edge_index, neg_edge_index):
        """
        Test reconstruction performance using AUC and AP metrics.
        
        Args:
            z: Gene embeddings
            pos_edge_index: Positive test edges
            neg_edge_index: Negative test edges
            
        Returns:
            auc: Area under ROC curve
            ap: Average precision score
        """
        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.ipd(z, pos_edge_index, sigmoid=True)
        neg_pred = self.ipd(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        return roc_auc_score(y, pred), average_precision_score(y, pred)

    def calculate_loss(self, x_foundation, x_raw, knn_edge_index, ppi_edge_index, highly_variable_index):
        """
        Calculate combined loss with foundation embeddings and raw expression data.
        
        Args:
            x_foundation: Foundation model embeddings (num_genes, foundation_embed_dim) 
            x_raw: Raw gene expression (num_genes, num_cells)
            knn_edge_index: KNN edges for cells
            ppi_edge_index: PPI edges for genes  
            highly_variable_index: HVG mask
            
        Returns:
            total_loss: Weighted combination of gene and cell losses
            row_loss: Gene reconstruction loss
            col_loss: Cell reconstruction loss
        """
        gene_emb = self.gene_encoder(x_foundation, x_raw, ppi_edge_index)
        cell_emb = self.cell_encoder(x_raw, knn_edge_index)
        
        cell_cross_emb, gene_cross_emb = self.cross_attention_encoder(cell_emb, gene_emb)
        
        max_cell_idx = cell_cross_emb.shape[0] - 1
        max_gene_idx = gene_cross_emb.shape[0] - 1
        
        valid_knn_mask = (knn_edge_index[0] <= max_cell_idx) & (knn_edge_index[1] <= max_cell_idx)
        filtered_knn_edge_index = knn_edge_index[:, valid_knn_mask]
        
        valid_ppi_mask = (ppi_edge_index[0] <= max_gene_idx) & (ppi_edge_index[1] <= max_gene_idx)
        filtered_ppi_edge_index = ppi_edge_index[:, valid_ppi_mask]
        
        embbed_rows = self.gene_final_encoder(gene_cross_emb, filtered_ppi_edge_index)
        row_loss = self.recon_loss(embbed_rows, filtered_ppi_edge_index, sig=True)
        
        embbed_cols = self.cell_final_encoder(cell_cross_emb, filtered_knn_edge_index)
        out_features = self.feature_decoder(embbed_cols)
        out_features = (out_features - (out_features.mean(axis=0))) / (out_features.std(axis=0) + EPS)
        reg = self.recon_loss(out_features.T, filtered_ppi_edge_index, sig=False)
        
        try:
            if hasattr(highly_variable_index, 'values'):
                hvg_mask = highly_variable_index.values
                hvg_tensor = torch.tensor(hvg_mask, dtype=torch.bool, device=x_raw.device)
            else:
                hvg_tensor = torch.tensor(highly_variable_index, dtype=torch.bool, device=x_raw.device)
            
            if hvg_tensor.shape[0] != x_raw.shape[0]:
                min_size = min(hvg_tensor.shape[0], x_raw.shape[0])
                hvg_tensor = hvg_tensor[:min_size]
                x_for_loss = x_raw[:min_size]
                out_features_for_loss = out_features[:, :min_size] if out_features.shape[1] > min_size else out_features
            else:
                x_for_loss = x_raw
                out_features_for_loss = out_features
            
            if hvg_tensor.sum() > 0:
                batch_size = out_features_for_loss.shape[0]
                x_hvg = x_for_loss[hvg_tensor]
                x_hvg_batch = x_hvg[:, :batch_size]
                out_features_hvg = out_features_for_loss.T[hvg_tensor].T
                col_loss = self.feature_criterion(x_hvg_batch.T, out_features_hvg)
            else:
                batch_size = out_features_for_loss.shape[0]
                x_batch = x_for_loss[:, :batch_size]
                col_loss = self.feature_criterion(x_batch.T, out_features_for_loss)
                
        except Exception as e:
            batch_size = out_features.shape[0]
            x_batch = x_raw[:, :batch_size]
            col_loss = self.feature_criterion(x_batch.T, out_features)
        
        return self.lambda_genes * row_loss + self.lambda_cells * (col_loss + reg), row_loss, col_loss
    
    def forward(self, x_foundation, x_raw, knn_edge_index, ppi_edge_index):
        """
        Forward pass with foundation embeddings and raw expression data.
        
        Args:
            x_foundation: Foundation model embeddings (num_genes, foundation_embed_dim)
            x_raw: Raw gene expression (num_genes, num_cells)
            knn_edge_index: KNN edges for cells
            ppi_edge_index: PPI edges for genes
            
        Returns:
            embedded_genes: Final gene embeddings
            embedded_cells: Final cell embeddings
            out_features: Reconstructed gene expression
        """
        gene_emb = self.gene_encoder(x_foundation, x_raw, ppi_edge_index)
        cell_emb = self.cell_encoder(x_raw, knn_edge_index)
        
        cell_cross_emb, gene_cross_emb = self.cross_attention_encoder(cell_emb, gene_emb)
        
        max_cell_idx = cell_cross_emb.shape[0] - 1
        max_gene_idx = gene_cross_emb.shape[0] - 1
        
        valid_knn_mask = (knn_edge_index[0] <= max_cell_idx) & (knn_edge_index[1] <= max_cell_idx)
        filtered_knn_edge_index = knn_edge_index[:, valid_knn_mask]
        
        valid_ppi_mask = (ppi_edge_index[0] <= max_gene_idx) & (ppi_edge_index[1] <= max_gene_idx)
        filtered_ppi_edge_index = ppi_edge_index[:, valid_ppi_mask]
        
        embedded_genes = self.gene_final_encoder(gene_cross_emb, filtered_ppi_edge_index)
        embedded_cells = self.cell_final_encoder(cell_cross_emb, filtered_knn_edge_index, infrance=True)
        out_features = self.feature_decoder(embedded_cells)

        return embedded_genes, embedded_cells, out_features