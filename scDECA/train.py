import torch
import torch.nn as nn
from tqdm import tqdm
import gc
import pkg_resources
import scDECA.utils as ut
from scDECA.attention import analyze_cross_attention_complete
import os


def evaluate_reconstruction(model, x_foundation, x_raw, data, knn_edge_index):
    """
    Evaluate model reconstruction performance on test edges.
    
    Args:
        model: scDECA model
        x_foundation: Foundation model embeddings (num_genes, foundation_embed_dim)
        x_raw: Raw gene expression (num_genes, num_cells)
        data: PyG Data object with test edges
        knn_edge_index: KNN edges for cells
    
    Returns:
        auc: Area under ROC curve
        ap: Average precision score
    """
    model.eval()
    with torch.no_grad():
        embbed_rows, _, _ = model(x_foundation, x_raw, knn_edge_index, data.train_pos_edge_index)
    return model.test(embbed_rows, data.test_pos_edge_index, data.test_neg_edge_index)



import torch
import torch.nn as nn
from tqdm import tqdm
import gc
import pkg_resources
from . import utils as ut
from .attention import analyze_cross_attention_complete
import os


def evaluate_reconstruction(model, x_foundation, x_raw, data, knn_edge_index):
    """
    Evaluate model reconstruction performance on test edges.
    
    Args:
        model: scDECA model
        x_foundation: Foundation model embeddings (num_genes, foundation_embed_dim)
        x_raw: Raw gene expression (num_genes, num_cells)
        data: PyG Data object with test edges
        knn_edge_index: KNN edges for cells
    
    Returns:
        auc: Area under ROC curve
        ap: Average precision score
    """
    model.eval()
    with torch.no_grad():
        embbed_rows, _, _ = model(x_foundation, x_raw, knn_edge_index, data.train_pos_edge_index)
    return model.test(embbed_rows, data.test_pos_edge_index, data.test_neg_edge_index)


def train(data, loader, highly_variable_index, x_raw, device, model_name, 
          number_of_batches=5, max_epoch=500, network_reduction_interval=30,
          cell_batching_flag=False, enable_attention_analysis=True, projection_dim=None,
          cell_names=None, gene_names=None, inter_dim=512, embedding_dim=258,
          num_layers=3, num_heads=4):
    """
    Main training loop for scDECA model.
    Supports both cell batching and full batch training modes.
    Performs final attention analysis after training completion.
    
    Args:
        data: PyG Data object with train/test edges
        loader: DataLoader for batching
        highly_variable_index: Mask for highly variable genes
        x_raw: Raw gene expression (num_genes, num_cells)
        device: Training device (cuda/cpu)
        model_name: Name for saving outputs
        number_of_batches: Number of batches for training
        max_epoch: Maximum training epochs
        network_reduction_interval: Epochs between network pruning
        cell_batching_flag: Whether to use cell batching mode
        enable_attention_analysis: Whether to analyze attention at end
        gene_attention_heads: Number of heads for gene attention (unused, kept for compatibility)
        projection_dim: Projection dimension for fusion
        cell_names: List of cell names for attention analysis
        gene_names: List of gene names for attention analysis
        inter_dim: Intermediate dimension
        embedding_dim: Output embedding dimension
        num_layers: Number of GNN layers
        num_heads: Number of attention heads
        
    Returns:
        model: Trained scDECA model
    """
    from .scDECA import scDECA
    
    x_foundation = data.x.clone()
    x_raw = x_raw.clone()
    
    if cell_batching_flag:
        cells_per_batch = x_raw.shape[1] // number_of_batches
        
        model = scDECA(cells_per_batch, x_raw.shape[0], x_foundation.shape[1],
                     inter_dim, embedding_dim, inter_dim, embedding_dim, 
                     lambda_genes=1, lambda_cells=1, num_layers=num_layers, 
                     num_heads=num_heads,
                     projection_dim=projection_dim).to(device)
    else:
        model = scDECA(x_raw.shape[1], x_raw.shape[0], x_foundation.shape[1],
                     inter_dim, embedding_dim, inter_dim, embedding_dim, 
                     lambda_genes=1, lambda_cells=1, num_layers=num_layers,
                     num_heads=num_heads,
                     projection_dim=projection_dim).to(device)
        
        x_raw_norm = ((x_raw.T - (x_raw.mean(axis=1))) / (x_raw.std(axis=1) + 0.00001)).T

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20)

    concat_flag = False
    
    # Create progress bar with postfix for metrics
    pbar = tqdm(range(max_epoch), desc="Training")

    for epoch in pbar:
        model.train()
        
        epoch_total_loss = 0.0
        epoch_row_loss = 0.0
        epoch_col_loss = 0.0
        batch_count = 0
        
        col_emb_lst = []
        row_emb_lst = []
        imput_lst = []
        batch_out_features = []
        normalized_x_lst = []

        for batch_idx, batch in enumerate(loader):
            if cell_batching_flag:
                batch_indices = batch[1]
                x_raw_batch = x_raw[:, batch_indices]
                x_raw_batch = ((x_raw_batch.T - (x_raw_batch.mean(axis=1))) / (x_raw_batch.std(axis=1) + 0.00001)).T
                from .data_loader import generate_knn_batch
                knn_edge_index = generate_knn_batch(loader.dataset.knn, batch_indices)
            else:
                x_raw_batch = x_raw_norm
                knn_edge_index = batch.T.to(device)

            if cell_batching_flag or knn_edge_index.shape[1] == loader.dataset.edge_index.shape[0] // number_of_batches:
                
                loss, row_loss, col_loss = model.calculate_loss(
                    x_foundation.to(device),
                    x_raw_batch.to(device),
                    knn_edge_index.to(device),
                    data.train_pos_edge_index, 
                    highly_variable_index
                )
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_total_loss += float(loss.item())
                epoch_row_loss += float(row_loss.item() if hasattr(row_loss, 'item') else row_loss)
                epoch_col_loss += float(col_loss.item() if hasattr(col_loss, 'item') else col_loss)
                batch_count += 1

                with torch.no_grad():
                    if cell_batching_flag:
                        row_embed, col_embed, out_features = model(
                            x_foundation.to(device),
                            x_raw_batch.to(device),
                            knn_edge_index, 
                            data.train_pos_edge_index
                        )
                        cell_input_emb = model.cell_encoder(x_raw_batch.to(device), knn_edge_index)
                        col_emb_lst.append(col_embed.cpu())
                        row_emb_lst.append(row_embed.cpu())
                        imput_lst.append(cell_input_emb.T.cpu())
                        batch_out_features.append(out_features.cpu())
                        normalized_x_lst.append(x_raw_batch.cpu())
                    else:
                        row_embed, col_embed, out_features = model(
                            x_foundation.to(device),
                            x_raw_batch.to(device),
                            knn_edge_index.to(device), 
                            data.train_pos_edge_index
                        )
            else:
                concat_flag = True
            
            gc.collect()
            torch.cuda.empty_cache()

        if batch_count > 0:
            avg_total_loss = float(epoch_total_loss / batch_count)
            avg_row_loss = float(epoch_row_loss / batch_count)
            avg_col_loss = float(epoch_col_loss / batch_count)
        else:
            avg_total_loss = avg_row_loss = avg_col_loss = 0.0

        if not cell_batching_flag:
            new_knn_edge_index, _ = model.cell_final_encoder.reduce_network()   

            if concat_flag:
                new_knn_edge_index = torch.concat([new_knn_edge_index, knn_edge_index], axis=-1)
                knn_edge_index = new_knn_edge_index

            if (epoch + 1) % network_reduction_interval == 0:
                from .data_loader import create_knn_loader
                loader = create_knn_loader(new_knn_edge_index, new_knn_edge_index.shape[1] // number_of_batches)

        if epoch % 10 == 0 or epoch == max_epoch - 1:
            if not cell_batching_flag:
                knn_edge_index = list(loader)[0].T.to(device)

            auc, ap = evaluate_reconstruction(model, x_foundation.to(device), x_raw_batch.to(device), data, knn_edge_index)
            
            scheduler.step(auc)

            # Update progress bar with metrics
            pbar.set_postfix({
                'Loss': f'{avg_total_loss:.4f}',
                'Row': f'{avg_row_loss:.4f}',
                'Col': f'{avg_col_loss:.4f}',
                'AUC': f'{auc:.4f}',
                'AP': f'{ap:.4f}'
            })

            package_dir = os.path.dirname(__file__)               
            models_root = os.path.join(package_dir, "Models")       
            model_dir = os.path.join(models_root, model_name)       

            embedding_dir      = os.path.join(model_dir, "Embeddings")
            node_features_dir  = os.path.join(model_dir, "Node_features")
            knn_dir            = os.path.join(model_dir, "KNN")

            os.makedirs(embedding_dir, exist_ok=True)
            os.makedirs(node_features_dir, exist_ok=True)
            os.makedirs(knn_dir, exist_ok=True)

            if cell_batching_flag:
                if row_emb_lst and col_emb_lst:
                    stacked_row_embed = torch.stack(row_emb_lst)
                    averaged_row_embed = stacked_row_embed.mean(dim=0)
                    concatenated_col_embed = torch.cat(col_emb_lst, dim=0)
                    
                    ut.save_obj(concatenated_col_embed.cpu().detach().numpy(), 
                            os.path.join(embedding_dir, "cell_embedding"))
                    ut.save_obj(averaged_row_embed.cpu().detach().numpy(), 
                            os.path.join(embedding_dir, "gene_embedding"))
                
                if batch_out_features:
                    concatenated_out_features = torch.cat(batch_out_features, dim=0)
                    ut.save_obj(concatenated_out_features.cpu().detach().numpy(),  
                            os.path.join(embedding_dir, "reconstructed_feature"))
                
                if normalized_x_lst:
                    concatenated_normalized_x = torch.cat(normalized_x_lst, dim=1)
                    ut.save_obj(concatenated_normalized_x.cpu().detach().numpy(),
                            os.path.join(embedding_dir, "normalized_feature"))
            else:
                ut.save_obj(new_knn_edge_index.cpu(), 
                        os.path.join(knn_dir, "knn_graph"))
                ut.save_obj(col_embed.cpu().detach().numpy(), 
                        os.path.join(embedding_dir, "cell_embedding"))
                ut.save_obj(row_embed.cpu().detach().numpy(), 
                        os.path.join(embedding_dir, "gene_embedding"))
                ut.save_obj(out_features.cpu().detach().numpy(),  
                        os.path.join(embedding_dir, "reconstructed_feature"))
                ut.save_obj(x_raw_batch.cpu().detach().numpy(),
                        os.path.join(embedding_dir, "normalized_feature"))
    
    if enable_attention_analysis and cell_names is not None and gene_names is not None:
        print("\n" + "="*60)
        print("PERFORMING FINAL ATTENTION ANALYSIS")
        print("="*60)
        
        # Use new organized structure
        package_dir = os.path.dirname(__file__)
        final_attention_dir = os.path.join(
            package_dir, "Models", model_name, "Attention", "final_analysis"
        )
        os.makedirs(final_attention_dir, exist_ok=True)
                
        try:
            if cell_batching_flag:
                batch_indices = batch[1]
                actual_cell_names = [cell_names[i] for i in batch_indices] if cell_names else None
                actual_x_raw = x_raw_batch
                actual_knn_edge_index = knn_edge_index
            else:
                actual_num_cells = x_raw_batch.shape[1]
                if cell_names and len(cell_names) > actual_num_cells:
                    actual_cell_names = cell_names[:actual_num_cells]
                elif cell_names and len(cell_names) < actual_num_cells:
                    actual_cell_names = cell_names + [f"Cell_{i}" for i in range(len(cell_names), actual_num_cells)]
                else:
                    actual_cell_names = cell_names
                actual_x_raw = x_raw_batch
                actual_knn_edge_index = knn_edge_index
            
            print(f"Final attention analysis dimensions:")
            print(f"   - Genes: {len(gene_names)}")
            print(f"   - Cells: {len(actual_cell_names) if actual_cell_names else 'None'}")
            print(f"   - x_raw shape: {actual_x_raw.shape}")
            
            final_analyzer = analyze_cross_attention_complete(
                model=model,
                x_foundation=x_foundation.to(device),
                x_raw=actual_x_raw.to(device),
                knn_edge_index=actual_knn_edge_index,
                ppi_edge_index=data.train_pos_edge_index,
                cell_names=actual_cell_names,
                gene_names=gene_names,
                output_dir=final_attention_dir
            )
            
            print(f"Final attention analysis completed and saved to: {final_attention_dir}")
            
        except Exception as e:
            print(f"Final attention analysis failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Save final model
    model_dir = os.path.join("Models", model_name)
    os.makedirs(model_dir, exist_ok=True)  

    model_save_path = os.path.join(model_dir, "model.pt")
    torch.save(model.state_dict(), model_save_path)
    
    return model