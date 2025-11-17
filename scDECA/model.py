import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import sequential, GATConv, GraphNorm, VGAE, GCNConv, InnerProductDecoder, TransformerConv, GAE, LayerNorm, SAGEConv
from torch_geometric.utils import negative_sampling, softmax
from sklearn.metrics import average_precision_score, roc_auc_score
import math

EPS = 1e-15
MAX_LOGSTD = 10


class ExpressionDecoder(torch.nn.Module):
    """
    Gene expression reconstruction decoder.
    Reconstructs gene expression from cell embeddings using GELU activation for smooth gradients.
    
    Args:
        feature_dim: Number of genes (output dimension)
        embd_dim: Embedding dimension (input dimension)
        inter_dim: Intermediate layer dimension
        drop_p: Dropout probability
    """
    def __init__(self, feature_dim, embd_dim, inter_dim, drop_p=0.0):
        super(ExpressionDecoder, self).__init__()
        self.feature_dim = feature_dim
        self.embd_dim = embd_dim
        self.inter_dim = inter_dim
        self.decoder = nn.Sequential(
            nn.Linear(embd_dim, inter_dim),
            nn.GELU(),  # GELU for smooth, stable reconstruction
            nn.Dropout(drop_p),
            nn.Linear(inter_dim, inter_dim),
            nn.GELU(),  # GELU suitable for continuous outputs
            nn.Dropout(drop_p),
            nn.Linear(inter_dim, feature_dim)
        )
              
    def forward(self, z):
        """
        Args:
            z: Cell embeddings (num_cells, embd_dim)
        Returns:
            Reconstructed gene expression (num_cells, num_genes)
        """
        out = self.decoder(z)
        return out


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism with dimension adjustment support.
    Uses Sigmoid for gates and keeps Softmax for attention weights.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.original_embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Store last attention weights for analysis
        self.last_attention_weights = None
        
        # Adjust embed_dim if not divisible by num_heads
        if embed_dim % num_heads != 0:
            adjusted_embed_dim = ((embed_dim // num_heads) + 1) * num_heads
            print(f"Warning: embed_dim {embed_dim} not divisible by num_heads {num_heads}")
            print(f"Adjusting embed_dim to {adjusted_embed_dim}")
            self.embed_dim = adjusted_embed_dim
            # Pure linear transformation (no non-linearity)
            self.input_proj = nn.Linear(embed_dim, adjusted_embed_dim)
            self.output_proj = nn.Linear(adjusted_embed_dim, embed_dim)
        else:
            self.embed_dim = embed_dim
            self.input_proj = None
            self.output_proj = None
            
        self.head_dim = self.embed_dim // num_heads
        
        self.q_linear = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_linear = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_linear = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_linear = nn.Linear(self.embed_dim, self.embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.original_embed_dim)
        
    def forward(self, query, key, value, mask=None):
        """
        Multi-head attention forward pass.
        
        Args:
            query: Query tensor (batch_size, seq_len, embed_dim)
            key: Key tensor (batch_size, seq_len, embed_dim)
            value: Value tensor (batch_size, seq_len, embed_dim)
            mask: Optional attention mask
            
        Returns:
            output: Attention output with residual connection
            attention_weights: Attention weight matrix
        """
        batch_size, seq_len, _ = query.size()
        
        # Store original query for residual connection
        original_query = query
        
        # Project to adjusted dimension if needed
        if self.input_proj is not None:
            query = self.input_proj(query)
            key = self.input_proj(key)
            value = self.input_proj(value)
        
        # Linear transformations and reshape for multi-head
        Q = self.q_linear(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, key.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, value.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention (keep Softmax - optimal for attention)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)  # Keep Softmax
        attention_weights = self.dropout(attention_weights)
        
        # Store attention weights for analysis
        self.last_attention_weights = attention_weights.detach()
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        output = self.out_linear(context)
        
        # Project back to original dimension if needed
        if self.output_proj is not None:
            output = self.output_proj(output)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + original_query)
        
        return output, attention_weights


class CrossAttentionEncoder(torch.nn.Module):
    """
    Cross-modal attention encoder for cell and gene representations.
    Enables bidirectional information flow between cell and gene embeddings.
    Uses Swish activation in FFN layers.
    
    Args:
        cell_dim: Cell embedding dimension
        gene_dim: Gene embedding dimension
        embed_dim: Shared embedding dimension for attention
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        dropout: Dropout probability
    """
    def __init__(self, cell_dim, gene_dim, embed_dim, num_heads=8, num_layers=2, dropout=0.1):
        super(CrossAttentionEncoder, self).__init__()
        self.cell_dim = cell_dim
        self.gene_dim = gene_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        # Store attention weights for analysis
        self.cell_cross_attention_weights = []
        self.gene_cross_attention_weights = []
        
        # Projection layers with Swish activation
        self.cell_proj = nn.Sequential(
            nn.Linear(cell_dim, embed_dim),
            nn.SiLU()  # Swish: self-gating for selective information flow
        )
        self.gene_proj = nn.Sequential(
            nn.Linear(gene_dim, embed_dim),
            nn.SiLU()  # Swish
        )
        
        # Self-attention layers for cell and gene
        self.cell_self_attention = nn.ModuleList([
            MultiHeadAttention(embed_dim, num_heads, dropout) 
            for _ in range(num_layers)
        ])
        
        self.gene_self_attention = nn.ModuleList([
            MultiHeadAttention(embed_dim, num_heads, dropout) 
            for _ in range(num_layers)
        ])
        
        # Cross-attention layers: cell attends to gene and vice versa
        self.cell_cross_attention = nn.ModuleList([
            MultiHeadAttention(embed_dim, num_heads, dropout) 
            for _ in range(num_layers)
        ])
        
        self.gene_cross_attention = nn.ModuleList([
            MultiHeadAttention(embed_dim, num_heads, dropout) 
            for _ in range(num_layers)
        ])
        
        # Feed-forward networks with Swish (optimal for FFN performance)
        self.cell_ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.SiLU(),  # Swish: best performance in FFN
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        
        self.gene_ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.SiLU(),  # Swish: best performance in FFN
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        
        # Separate LayerNorm for cell and gene
        self.cell_ln = nn.LayerNorm(embed_dim)
        self.gene_ln = nn.LayerNorm(embed_dim)
        
        # Skip connection weight
        self.skip_weight = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, cell_features, gene_features):
        """
        Forward pass with bidirectional cross-attention.
        
        Args:
            cell_features: Cell embeddings (num_cells, cell_dim)
            gene_features: Gene embeddings (num_genes, gene_dim)
            
        Returns:
            cell_emb: Enhanced cell embeddings (num_cells, embed_dim)
            gene_emb: Enhanced gene embeddings (num_genes, embed_dim)
        """
        # Project to shared embedding space and add batch dimension
        cell_emb = self.cell_proj(cell_features).unsqueeze(0)  # [1, num_cells, embed_dim]
        gene_emb = self.gene_proj(gene_features).unsqueeze(0)  # [1, num_genes, embed_dim]
        
        # Initialize attention weight storage
        self.cell_cross_attention_weights = []
        self.gene_cross_attention_weights = []
        
        for i in range(self.num_layers):
            # Self-attention: attend to same modality
            cell_self_out, _ = self.cell_self_attention[i](cell_emb, cell_emb, cell_emb)
            gene_self_out, _ = self.gene_self_attention[i](gene_emb, gene_emb, gene_emb)
            
            # Cross-attention: attend to other modality
            cell_cross_out, cell_cross_attn = self.cell_cross_attention[i](cell_self_out, gene_self_out, gene_self_out)
            gene_cross_out, gene_cross_attn = self.gene_cross_attention[i](gene_self_out, cell_self_out, cell_self_out)
            
            # Residual connection
            cell_cross_out = cell_cross_out + cell_self_out
            gene_cross_out = gene_cross_out + gene_self_out
            
            # Store cross-attention weights
            self.cell_cross_attention_weights.append(cell_cross_attn.detach())
            self.gene_cross_attention_weights.append(gene_cross_attn.detach())
            
            # Feed-forward network
            cell_ffn_out = self.cell_ffn[i](cell_cross_out)
            gene_ffn_out = self.gene_ffn[i](gene_cross_out)
            
            # Residual connection and layer norm with separate LayerNorm
            cell_emb = self.cell_ln(cell_cross_out + cell_ffn_out)
            gene_emb = self.gene_ln(gene_cross_out + gene_ffn_out)
        
        return cell_emb.squeeze(0), gene_emb.squeeze(0)
    
    def get_cell_cross_attention_weights(self, layer_idx=None):
        if layer_idx is None:
            return self.cell_cross_attention_weights
        else:
            return self.cell_cross_attention_weights[layer_idx] if layer_idx < len(self.cell_cross_attention_weights) else None
    
    def get_gene_cross_attention_weights(self, layer_idx=None):
        if layer_idx is None:
            return self.gene_cross_attention_weights
        else:
            return self.gene_cross_attention_weights[layer_idx] if layer_idx < len(self.gene_cross_attention_weights) else None


class CellEncoder(torch.nn.Module):
    """
    Cell-level GNN encoder using KNN graph.
    Uses ELU activation for better gradient flow in graph structures.
    
    Args:
        input_dim: Number of genes (features per cell)
        embed_dim: Output embedding dimension
        num_layers: Number of graph convolution layers
        drop_p: Dropout probability
    """
    def __init__(self, input_dim, embed_dim, num_layers=2, drop_p=0.25):
        super(CellEncoder, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        # Cell encoder using KNN network with ELU (stable for graph structures)
        self.layers = nn.ModuleList([
            sequential.Sequential('x,edge_index', [
                (SAGEConv(input_dim if i == 0 else embed_dim, embed_dim), 'x, edge_index -> x1'),
                nn.ELU(inplace=True),  # Conv → ELU → Dropout order
                (nn.Dropout(drop_p, inplace=False), 'x1 -> x2'),
            ]) for i in range(num_layers)
        ])
        
    def forward(self, x, knn_edge_index):
        """
        Forward pass through cell encoder.
        
        Args:
            x: Gene expression matrix (num_genes, num_cells)
            knn_edge_index: KNN graph edges
            
        Returns:
            Cell embeddings (num_cells, embed_dim)
        """
        # Transpose: (num_genes, num_cells) -> (num_cells, num_genes)
        cell_features = x.T
        cell_emb = cell_features.clone()
        for layer in self.layers:
            cell_emb = layer(cell_emb, knn_edge_index)
        
        return cell_emb


class GeneEncoder(torch.nn.Module):
    """
    Gene-level encoder with foundation model fusion.
    Integrates foundation model embeddings with raw expression data using addition fusion.
    Uses mixed activations optimized for different components.
    
    Args:
        scgpt_embed_dim: Foundation model embedding dimension
        expression_dim: Raw expression dimension (number of cells)
        embed_dim: Output embedding dimension
        num_layers: Number of graph convolution layers
        drop_p: Dropout probability
        projection_dim: Intermediate projection dimension
    """
    def __init__(self, scgpt_embed_dim, expression_dim, embed_dim, num_layers=2, drop_p=0.25, 
                 projection_dim=None):
        super(GeneEncoder, self).__init__()
        self.scgpt_embed_dim = scgpt_embed_dim
        self.expression_dim = expression_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        if projection_dim is None:
            projection_dim = embed_dim
        self.intermediate_dim = projection_dim
        
        print(f"GeneEncoder (add fusion): SAGEConv→{self.intermediate_dim}, FM→{self.intermediate_dim}, fusion→{embed_dim}")
        
        # Step 1: Raw expression → SAGEConv with ELU (stable for graph structures)
        self.expression_layers = nn.ModuleList([
            sequential.Sequential('x,edge_index', [
                (SAGEConv(expression_dim if i == 0 else self.intermediate_dim, self.intermediate_dim), 'x, edge_index -> x1'),
                nn.ELU(inplace=True),  # Conv → ELU → Dropout order
                (nn.Dropout(drop_p, inplace=False), 'x1 -> x2'),
            ]) for i in range(num_layers)
        ])
        
        # Step 2: Foundation embedding projection with GELU (compatible with pretrained models)
        self.foundation_proj = nn.Sequential(
            nn.Linear(scgpt_embed_dim, self.intermediate_dim),
            nn.LayerNorm(self.intermediate_dim),
            nn.GELU(),  # GELU: good compatibility with pretrained Transformer models
            nn.Dropout(drop_p)
        )
        
        # Step 3: Fusion layer (add fusion only)
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.intermediate_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),  # Swish
            nn.Dropout(drop_p)
        )
        
        # Step 4: Final refinement layers with ELU (graph structure refinement)
        self.final_layers = nn.ModuleList([
            sequential.Sequential('x,edge_index', [
                (SAGEConv(embed_dim, embed_dim), 'x, edge_index -> x1'),
                nn.ELU(inplace=True),  # Conv → ELU → Dropout order
                (nn.Dropout(drop_p, inplace=False), 'x1 -> x2'),
            ]) for i in range(max(1, num_layers//2))
        ])
        
    def forward(self, x_foundation, x_expression, ppi_edge_index):
        """
        Forward pass fusing foundation embeddings and expression data using addition.
        
        Args:
            x_foundation: Foundation model embeddings (num_genes, foundation_dim)
            x_expression: Raw gene expression (num_genes, num_cells)
            ppi_edge_index: PPI graph edges
            
        Returns:
            Gene embeddings (num_genes, embed_dim)
        """
        # Step 1: Raw expression → SAGEConv → intermediate_dim
        expression_processed = x_expression.clone()
        for layer in self.expression_layers:
            expression_processed = layer(expression_processed, ppi_edge_index)
        
        # Step 2: Foundation embedding projection → intermediate_dim
        foundation_projected = self.foundation_proj(x_foundation)
        
        # Ensure gene count matches
        assert expression_processed.shape[0] == foundation_projected.shape[0], \
            f"Gene count mismatch: {expression_processed.shape[0]} != {foundation_projected.shape[0]}"
        
        # Ensure dimensions match for addition
        assert expression_processed.shape[1] == foundation_projected.shape[1], \
            f"Dimension mismatch for add fusion: {expression_processed.shape[1]} != {foundation_projected.shape[1]}"
        
        # Step 3: Addition fusion
        fused_features = expression_processed + foundation_projected
        fused_output = self.fusion_layer(fused_features)
        
        # Step 4: Final refinement
        gene_emb = fused_output.clone()
        for layer in self.final_layers:
            gene_emb = layer(gene_emb, ppi_edge_index)
        
        return gene_emb


class AdaptiveGraphConvLayer(TransformerConv):
    """
    Adaptive graph convolution layer with attention-based edge pruning.
    Extends TransformerConv to support dynamic network reduction.
    
    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension
        heads: Number of attention heads
        dropout: Dropout probability
        add_self_loops: Whether to add self-loops
        scale_param: Scaling parameter for attention normalization
    """
    def __init__(self, in_channels, out_channels, heads=1, dropout=0, add_self_loops=True, scale_param=2, **kwargs):
        super().__init__(in_channels, out_channels, heads, dropout, add_self_loops, **kwargs)
        self.treshold_alpha = None
        self.scale_param = scale_param
    
    def message(self, query_i, key_j, value_j, edge_attr, index, ptr, size_i):
        """Custom message passing with adaptive attention."""
        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
            key_j += edge_attr

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        if not self.scale_param is None:
            alpha = alpha - alpha.mean()
            alpha = alpha / ((1/self.scale_param) * alpha.std())
            alpha = F.sigmoid(alpha)
        else:
            alpha = softmax(alpha, index, ptr, size_i)
        self.treshold_alpha = alpha 

        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j
        if edge_attr is not None:
            out += edge_attr

        out *= alpha.view(-1, self.heads, 1)
        return out


class StructuralEncoder(torch.nn.Module):
    """
    Structural encoder for graph representation learning.
    Supports dynamic network reduction based on attention weights.
    Uses Mish activation for high performance.
    
    Args:
        feature_dim: Input feature dimension
        inter_dim: Intermediate dimension
        embd_dim: Output embedding dimension
        reducer: Whether to enable network reduction
        drop_p: Dropout probability
        scale_param: Scaling parameter for attention
    """
    def __init__(self, feature_dim, inter_dim, embd_dim, reducer=False, drop_p=0.2, scale_param=3):
        super(StructuralEncoder, self).__init__()
        self.feature_dim = feature_dim
        self.embd_dim = embd_dim
        self.inter_dim = inter_dim
        self.reducer = reducer

        # Mish activation: state-of-the-art activation function
        self.mish = nn.Mish()

        self.encoder = sequential.Sequential('x, edge_index', [
            (GCNConv(self.feature_dim, self.inter_dim), 'x, edge_index -> x1'),
            self.mish,  # Conv → Mish → Dropout order
            (nn.Dropout(drop_p, inplace=False), 'x1-> x2')
        ])
        
        if self.reducer:                
            self.atten_layer = AdaptiveGraphConvLayer(self.inter_dim, self.embd_dim, dropout=drop_p, add_self_loops=False, heads=1, scale_param=scale_param)
        else:
            self.atten_layer = TransformerConv(self.inter_dim, self.embd_dim, dropout=drop_p)
           
        self.atten_map = None
        self.atten_weights = None
        self.plot_count = 0

    def reduce_network(self, threshold=0.2, min_connect=10):
        """
        Reduce network based on attention weights.
        Keeps top-k edges per node and edges above threshold.
        """
        import pandas as pd
        import numpy as np
        
        self.plot_count += 1
        graph = self.atten_weights.cpu().detach().numpy()
        threshold_bound = np.percentile(graph, 10)
        threshold = min(threshold, threshold_bound) 
        df = pd.DataFrame({"v1": self.atten_map[0].cpu().detach().numpy(), "v2": self.atten_map[1].cpu().detach().numpy(), "atten": graph.squeeze()})
        saved_edges = df.groupby('v1')['atten'].nlargest(min_connect).index.values
        saved_edges = [v2 for _, v2 in saved_edges]
        df.iloc[saved_edges, 2] = threshold + EPS
        indexs = list(df.loc[df.atten >= threshold].index)
        atten_map = self.atten_map[:, indexs]
        self.atten_map = None
        self.atten_weights = None
        return atten_map, df 

    def forward(self, x, edge_index, infrance=False):
        """
        Forward pass through structural encoder.
        
        Args:
            x: Input features
            edge_index: Graph edges
            infrance: Inference mode (disables attention collection)
            
        Returns:
            Embedded features
        """
        embbded = x.clone()
        embbded = self.encoder(embbded, edge_index)
        embbded, atten_map = self.atten_layer(embbded, edge_index, return_attention_weights=True)
        if self.reducer and not infrance:
            if self.atten_map is None:
                self.atten_map = atten_map[0].detach()
                self.atten_weights = atten_map[1].detach()
            else:
                self.atten_map = torch.concat([self.atten_map.T, atten_map[0].detach().T]).T
                self.atten_weights = torch.concat([self.atten_weights, atten_map[1].detach()])

        return embbded