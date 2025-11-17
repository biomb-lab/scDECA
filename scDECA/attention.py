import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import os
warnings.filterwarnings('ignore')


class attention:
    """
    Analyzer for cross-attention patterns between cells and genes.
    Extracts and visualizes attention weights from trained scDECA model.
    """
    
    def __init__(self, model, cell_names=None, gene_names=None):
        """
        Args:
            model: Trained scDECA model
            cell_names: List of cell names (optional)
            gene_names: List of gene names (optional)
        """
        self.model = model
        self.cell_names = cell_names
        self.gene_names = gene_names
        self.attention_weights = {}
        
    def extract_attention_weights(self, x_foundation, x_raw, knn_edge_index, ppi_edge_index):
        """
        Extract cross-attention weights from model.
        
        Args:
            x_foundation: Foundation model embeddings
            x_raw: Raw gene expression
            knn_edge_index: KNN edges for cells
            ppi_edge_index: PPI edges for genes
            
        Returns:
            Dictionary containing cell and gene cross-attention weights
        """
        self.model.eval()
        with torch.no_grad():
            gene_emb = self.model.gene_encoder(x_foundation, x_raw, ppi_edge_index)
            cell_emb = self.model.cell_encoder(x_raw, knn_edge_index)
            
            cell_cross_emb, gene_cross_emb = self.model.cross_attention_encoder(cell_emb, gene_emb)
            
            self.attention_weights = {
                'cell_cross_attention': self.model.cross_attention_encoder.get_cell_cross_attention_weights(),
                'gene_cross_attention': self.model.cross_attention_encoder.get_gene_cross_attention_weights()
            }

        return self.attention_weights
    
    def analyze_cell_to_gene_attention(self, layer_idx=-1, top_k=10):
        """
        Analyze Cell → Gene attention: which genes each cell focuses on.
        
        Args:
            layer_idx: Layer index to analyze (-1 for last layer)
            top_k: Number of top genes to return
            
        Returns:
            cell_gene_attention: Dictionary mapping cells to top genes
            avg_attention: Average attention matrix (num_cells, num_genes)
        """
        if 'cell_cross_attention' not in self.attention_weights:
            raise ValueError("Run extract_attention_weights() first.")
        
        cell_attn_weights = self.attention_weights['cell_cross_attention'][layer_idx]
        avg_attention = cell_attn_weights.squeeze(0).mean(dim=0)
        
        cell_gene_attention = {}
        
        for cell_idx in range(avg_attention.shape[0]):
            cell_attention = avg_attention[cell_idx]
            top_gene_indices = torch.topk(cell_attention, k=min(top_k, len(cell_attention))).indices
            top_gene_scores = torch.topk(cell_attention, k=min(top_k, len(cell_attention))).values
            
            cell_name = self.cell_names[cell_idx] if self.cell_names else f"Cell_{cell_idx}"
            
            top_genes = []
            for i, (gene_idx, score) in enumerate(zip(top_gene_indices, top_gene_scores)):
                gene_name = self.gene_names[gene_idx] if self.gene_names else f"Gene_{gene_idx}"
                top_genes.append({
                    'gene_name': gene_name,
                    'gene_idx': gene_idx.item(),
                    'attention_score': score.item(),
                    'rank': i + 1
                })
            
            cell_gene_attention[cell_name] = top_genes
        
        return cell_gene_attention, avg_attention
    
    def analyze_gene_to_cell_attention(self, layer_idx=-1, top_k=10):
        """
        Analyze Gene → Cell attention: which cells each gene focuses on.
        
        Args:
            layer_idx: Layer index to analyze (-1 for last layer)
            top_k: Number of top cells to return
            
        Returns:
            gene_cell_attention: Dictionary mapping genes to top cells
            avg_attention: Average attention matrix (num_genes, num_cells)
        """
        if 'gene_cross_attention' not in self.attention_weights:
            raise ValueError("Run extract_attention_weights() first.")
        
        gene_attn_weights = self.attention_weights['gene_cross_attention'][layer_idx]
        avg_attention = gene_attn_weights.squeeze(0).mean(dim=0)
        
        gene_cell_attention = {}
        
        for gene_idx in range(avg_attention.shape[0]):
            gene_attention = avg_attention[gene_idx]
            top_cell_indices = torch.topk(gene_attention, k=min(top_k, len(gene_attention))).indices
            top_cell_scores = torch.topk(gene_attention, k=min(top_k, len(gene_attention))).values
            
            gene_name = self.gene_names[gene_idx] if self.gene_names else f"Gene_{gene_idx}"
            
            top_cells = []
            for i, (cell_idx, score) in enumerate(zip(top_cell_indices, top_cell_scores)):
                cell_name = self.cell_names[cell_idx] if self.cell_names else f"Cell_{cell_idx}"
                top_cells.append({
                    'cell_name': cell_name,
                    'cell_idx': cell_idx.item(),
                    'attention_score': score.item(),
                    'rank': i + 1
                })
            
            gene_cell_attention[gene_name] = top_cells
        
        return gene_cell_attention, avg_attention
    
    def save_full_attention_matrices(self, output_dir, layer_idx=-1):
        """
        Save complete attention matrices to CSV files (no subsampling).
        
        Args:
            output_dir: Directory to save matrices
            layer_idx: Layer index to save (-1 for last layer)
            
        Returns:
            cell_gene_df: Cell→Gene attention DataFrame
            gene_cell_df: Gene→Cell attention DataFrame
        """
        if 'cell_cross_attention' not in self.attention_weights or 'gene_cross_attention' not in self.attention_weights:
            raise ValueError("Run extract_attention_weights() first.")
        
        os.makedirs(output_dir, exist_ok=True)
        
        cell_attn_weights = self.attention_weights['cell_cross_attention'][layer_idx]
        cell_gene_matrix = cell_attn_weights.squeeze(0).mean(dim=0)
        
        actual_num_cells = cell_gene_matrix.shape[0]
        actual_num_genes = cell_gene_matrix.shape[1]
        
        print(f"Attention matrix dimensions: cells={actual_num_cells}, genes={actual_num_genes}")
        
        if self.cell_names is not None and len(self.cell_names) != actual_num_cells:
            print(f"Warning: cell_names length ({len(self.cell_names)}) != actual cells ({actual_num_cells})")
            if len(self.cell_names) > actual_num_cells:
                cell_names_used = self.cell_names[:actual_num_cells]
            else:
                cell_names_used = self.cell_names + [f"Cell_{i}" for i in range(len(self.cell_names), actual_num_cells)]
        else:
            cell_names_used = self.cell_names if self.cell_names else [f"Cell_{i}" for i in range(actual_num_cells)]
        
        if self.gene_names is not None and len(self.gene_names) != actual_num_genes:
            print(f"Warning: gene_names length ({len(self.gene_names)}) != actual genes ({actual_num_genes})")
            if len(self.gene_names) > actual_num_genes:
                gene_names_used = self.gene_names[:actual_num_genes]
            else:
                gene_names_used = self.gene_names + [f"Gene_{i}" for i in range(len(self.gene_names), actual_num_genes)]
        else:
            gene_names_used = self.gene_names if self.gene_names else [f"Gene_{i}" for i in range(actual_num_genes)]
        
        cell_gene_df = pd.DataFrame(
            cell_gene_matrix.detach().cpu().numpy(),
            index=cell_names_used,
            columns=gene_names_used
        )
        
        cell_gene_path = os.path.join(output_dir, "full_cell_to_gene_attention_matrix.csv")
        cell_gene_df.to_csv(cell_gene_path)
        
        gene_attn_weights = self.attention_weights['gene_cross_attention'][layer_idx]
        gene_cell_matrix = gene_attn_weights.squeeze(0).mean(dim=0)
        
        gene_cell_df = pd.DataFrame(
            gene_cell_matrix.detach().cpu().numpy(),
            index=gene_names_used,
            columns=cell_names_used
        )
        
        gene_cell_path = os.path.join(output_dir, "full_gene_to_cell_attention_matrix.csv")
        gene_cell_df.to_csv(gene_cell_path)
        
        return cell_gene_df, gene_cell_df
    
    def plot_attention_heatmap_sample(self, attention_matrix, title, cell_names=None, gene_names=None, 
                                    figsize=(12, 8), save_path=None, sample_size=50):
        """
        Visualize attention matrix as heatmap with subsampling for large matrices.
        
        Args:
            attention_matrix: Attention matrix to visualize
            title: Plot title
            cell_names: Cell names for labels
            gene_names: Gene names for labels
            figsize: Figure size
            save_path: Path to save plot
            sample_size: Number of top items to display
        """
        plt.figure(figsize=figsize)
        
        if attention_matrix.shape[0] > sample_size or attention_matrix.shape[1] > sample_size:
            
            row_sums = attention_matrix.sum(dim=1)
            col_sums = attention_matrix.sum(dim=0)
            
            top_rows = torch.topk(row_sums, k=min(sample_size, len(row_sums))).indices
            top_cols = torch.topk(col_sums, k=min(sample_size, len(col_sums))).indices
            
            attention_matrix_sample = attention_matrix[top_rows][:, top_cols]
            
            if cell_names and gene_names:
                if attention_matrix.shape[0] == len(row_sums):
                    cell_names_sample = [cell_names[i] for i in top_rows]
                    gene_names_sample = [gene_names[i] for i in top_cols]
                else:
                    gene_names_sample = [gene_names[i] for i in top_rows]
                    cell_names_sample = [cell_names[i] for i in top_cols]
            else:
                cell_names_sample = gene_names_sample = None
        else:
            attention_matrix_sample = attention_matrix
            cell_names_sample = cell_names
            gene_names_sample = gene_names
        
        attention_np = attention_matrix_sample.detach().cpu().numpy()
        
        if attention_matrix_sample.shape[0] <= attention_matrix_sample.shape[1]:
            x_labels = gene_names_sample[:attention_matrix_sample.shape[1]] if gene_names_sample else None
            y_labels = cell_names_sample[:attention_matrix_sample.shape[0]] if cell_names_sample else None
        else:
            x_labels = cell_names_sample[:attention_matrix_sample.shape[1]] if cell_names_sample else None
            y_labels = gene_names_sample[:attention_matrix_sample.shape[0]] if gene_names_sample else None
        
        sns.heatmap(attention_np, 
                   xticklabels=x_labels, 
                   yticklabels=y_labels,
                   cmap='Blues', 
                   cbar_kws={'label': 'Attention Weight'})
        
        plt.title(f"{title} (Top {sample_size} sample)")
        plt.xlabel('Genes' if attention_matrix_sample.shape[0] <= attention_matrix_sample.shape[1] else 'Cells')
        plt.ylabel('Cells' if attention_matrix_sample.shape[0] <= attention_matrix_sample.shape[1] else 'Genes')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_attention_analysis(self, output_dir, cell_gene_attention=None, gene_cell_attention=None):
        """
        Save attention analysis results to CSV files.
        
        Args:
            output_dir: Directory to save results
            cell_gene_attention: Cell→Gene attention dictionary
            gene_cell_attention: Gene→Cell attention dictionary
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if cell_gene_attention:
            cell_gene_df = []
            for cell_name, genes in cell_gene_attention.items():
                for gene_info in genes:
                    cell_gene_df.append({
                        'cell_name': cell_name,
                        'gene_name': gene_info['gene_name'],
                        'gene_idx': gene_info['gene_idx'],
                        'attention_score': gene_info['attention_score'],
                        'rank': gene_info['rank']
                    })
            
            df = pd.DataFrame(cell_gene_df)
            df.to_csv(output_dir / 'cell_to_gene_attention.csv', index=False)
            print(f"Saved Cell→Gene attention: {output_dir / 'cell_to_gene_attention.csv'}")
        
        if gene_cell_attention:
            gene_cell_df = []
            for gene_name, cells in gene_cell_attention.items():
                for cell_info in cells:
                    gene_cell_df.append({
                        'gene_name': gene_name,
                        'cell_name': cell_info['cell_name'],
                        'cell_idx': cell_info['cell_idx'],
                        'attention_score': cell_info['attention_score'],
                        'rank': cell_info['rank']
                    })
            
            df = pd.DataFrame(gene_cell_df)
            df.to_csv(output_dir / 'gene_to_cell_attention.csv', index=False)
            print(f"Saved Gene→Cell attention: {output_dir / 'gene_to_cell_attention.csv'}")


def analyze_cross_attention_complete(model, x_foundation, x_raw, knn_edge_index, ppi_edge_index,
                                   cell_names=None, gene_names=None, output_dir="attention_analysis"):
    """
    Complete cross-attention analysis pipeline.
    Extracts, analyzes, and visualizes attention patterns between cells and genes.
    
    Args:
        model: Trained scDECA model
        x_foundation: Foundation model embeddings
        x_raw: Raw gene expression
        knn_edge_index: KNN edges for cells
        ppi_edge_index: PPI edges for genes
        cell_names: List of cell names
        gene_names: List of gene names
        output_dir: Directory to save results
        
    Returns:
        analyzer: CrossAttentionAnalyzer instance with extracted weights
    """
    os.makedirs(output_dir, exist_ok=True)
    
    analyzer = attention(model, cell_names, gene_names)
    
    attention_weights = analyzer.extract_attention_weights(x_foundation, x_raw, knn_edge_index, ppi_edge_index)
    
    cell_gene_attention, cell_gene_matrix = analyzer.analyze_cell_to_gene_attention(top_k=15)
    
    gene_cell_attention, gene_cell_matrix = analyzer.analyze_gene_to_cell_attention(top_k=15)
    
    cell_gene_df, gene_cell_df = analyzer.save_full_attention_matrices(output_dir)
    
    analyzer.plot_attention_heatmap_sample(
        cell_gene_matrix, 
        "Cell → Gene Cross Attention",
        cell_names, gene_names,
        save_path=os.path.join(output_dir, "cell_to_gene_heatmap_sample.png")
    )
    
    analyzer.plot_attention_heatmap_sample(
        gene_cell_matrix,
        "Gene → Cell Cross Attention", 
        gene_names, cell_names,
        save_path=os.path.join(output_dir, "gene_to_cell_heatmap_sample.png")
    )
    
    analyzer.save_attention_analysis(output_dir, cell_gene_attention, gene_cell_attention)
    
    return analyzer