# simplified_embedding_extractor_scGPT_only.py

import os
import json
import warnings
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import scanpy as sc
import torch

# scGPT
import scgpt as scg
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.model import TransformerModel

warnings.filterwarnings("ignore")

# scGPT settings
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
pad_value = -2
n_bins = 51
n_input_bins = n_bins


# scGPT model loader
def load_scgpt_model(model_dir: str):
    # Load pretrained scGPT model
    print(f"[scGPT] Loading from {model_dir}")

    mdir = Path(model_dir)
    model_config_file = mdir / "args.json"
    model_file = mdir / "best_model.pt"
    vocab_file = mdir / "vocab.json"

    for fp in [model_config_file, model_file, vocab_file]:
        if not fp.exists():
            raise FileNotFoundError(f"Missing required file: {fp}")

    vocab = GeneVocab.from_file(vocab_file)
    for t in special_tokens:
        if t not in vocab:
            vocab.append_token(t)

    with open(model_config_file, "r") as f:
        cfg = json.load(f)

    embsize = cfg["embsize"]
    nhead = cfg["nheads"]
    d_hid = cfg["d_hid"]
    nlayers = cfg["nlayers"]

    gene2idx = vocab.get_stoi()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ntokens = len(vocab)
    model = TransformerModel(
        ntokens,
        embsize,
        nhead,
        d_hid,
        nlayers,
        vocab=vocab,
        pad_value=pad_value,
        n_input_bins=n_input_bins,
    )

    try:
        state = torch.load(model_file, map_location=device)
        model.load_state_dict(state)
        print("[scGPT] Loaded full weights")
    except Exception as e:
        print(f"[scGPT] Partial load: {e}")
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file, map_location=device)
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model.to(device).eval()
    print(f"[scGPT] OK | vocab={ntokens} | dim={embsize}")

    return model, vocab, gene2idx



# scGPT embedding creator
def create_scgpt_gene_embeddings(scgpt_model, gene2idx: Dict[str, int], target_genes):
    # Extract gene embeddings
    print("[scGPT] Creating gene embeddings ...")

    device = next(scgpt_model.parameters()).device
    available = [g for g in target_genes if g in gene2idx]
    missing = [g for g in target_genes if g not in gene2idx]

    print(f"[scGPT] Found {len(available)} genes; missing {len(missing)}")

    if not available:
        return None, [], missing

    gene_ids = np.array([gene2idx[g] for g in available])

    with torch.no_grad():
        emb = scgpt_model.encoder(
            torch.tensor(gene_ids, dtype=torch.long, device=device)
        )
        emb = emb.detach().cpu().numpy()

    df = pd.DataFrame(
        emb,
        index=available,
        columns=[f"scgpt_dim_{i}" for i in range(emb.shape[1])]
    )
    print(f"[scGPT] Shape: {df.shape}")

    return df, available, missing


# Public API (scGPT only)
def add_gene_embeddings_to_adata(
    adata: sc.AnnData,
    scgpt_model_dir: str,
) -> sc.AnnData:
    """
    Add scGPT gene embeddings to adata.varm.
    """
    print("==> Adding scGPT gene embeddings")
    all_genes = adata.var.index.tolist()
    print(f"   - Total genes: {len(all_genes)}")

    # Load scGPT
    print("\n=== scGPT ===")
    scgpt_model, vocab, gene2idx = load_scgpt_model(scgpt_model_dir)

    # Extract embeddings
    df_scg, avail_scg, miss_scg = create_scgpt_gene_embeddings(
        scgpt_model, gene2idx, all_genes
    )

    if df_scg is not None:
        common = sorted(set(df_scg.index) & set(all_genes))
        df_scg = df_scg.loc[common]
        df_scg = df_scg.reindex(all_genes, fill_value=np.nan)

        adata.varm["scGPT_gene_token"] = df_scg.values
        adata.uns["scGPT_gene_token_info"] = {
            "embedding_dim": df_scg.shape[1],
            "total_genes": len(all_genes),
            "genes_with_embeddings": len(common),
            "missing_genes": len(miss_scg),
        }

        print(f"[scGPT] Stored: {df_scg.shape}")

    print("==> Done")
    return adata