# **scDECA: Integrating Gene-Cell Interactions with Global Priors and Local Structures in Single-Cell Transcriptomics**

## **Overview**

Single-cell RNA sequencing enables high-resolution profiling of cellular transcriptomes but remains challenging due to **high dimensionality**, **sparsity**, and **dropout noise**.  
Single-cell foundation models capture **global gene-level semantics**, but often miss **condition-specific variation**.  
Conversely, graph-based GNN approaches recover **local relational structure** without leveraging broader biological context.

To address these limitations, we introduce **scDECA**, a **Dual-Encoder Cross-Attention** framework that integrates:

- **Gene Encoder** — FM-based gene embeddings + PPI graph  
- **Cell Encoder** — KNN similarity graph + contextual attention  
- **Cross-Attention Alignment** — aligns molecular and cellular representations  
- **Unified Gene–Cell Embeddings** — capture global semantics + local heterogeneity  

scDECA yields **biologically coherent**, **dropout-robust**, and **relation-aware** embeddings that significantly improve reconstruction, co-expression recovery, and pathway detection.

![Overview of the scDECA](https://github.com/user-attachments/assets/2f1b1f5a-6c74-43c5-b07a-c6ce9b2d03b5)

## 📦 **1. Environment Setup**

### **Create and activate a clean Python environment**
```bash
conda create -n scDECA python=3.10 -y
conda activate scDECA
```
### **Install the remaining Python dependencies**
```bash
pip install -r requirements.txt
```

## 🔧 **2. Download Pretrained Models**

### **scGPT Pretrained Models**
Before extracting gene embeddings, you need to download the pretrained scGPT models:


Download the desired model from the [scGPT Model Zoo](https://github.com/bowang-lab/scGPT/tree/main):
   - Visit the Pretrained scGPT Model Zoo section
   - Choose the appropriate model for your analysis
   - Download and place it in `scDECA/pretrained/`

For example, to use the whole-human model:
```bash
# Download the model file to scDECA/pretrained/
cd scDECA/pretrained
# Follow the download instructions from the scGPT repository
```

**Note**: Make sure the model file is properly placed in `scDECA/pretrained/` before proceeding to gene embedding extraction.

## 🧬 **3. Single-cell Foundation Models Gene Token Embedding Extraction**

scDECA requires pretrained gene embeddings stored in `adata.varm`. We provide a complete extraction pipeline using scGPT.

**Important**: You must complete this step before running scDECA, as the model depends on these pretrained gene-level representations.

See: **[FMs Gene Embedding Extraction.ipynb](https://github.com/biomb-lab/scDECA/blob/main/embedding_extract.ipynb)**

## 🚀 **4. Running scDECA**

### **API Usage**

Train scDECA using your AnnData object with pretrained gene embeddings:

```python
import scDECA

# Run scDECA with default parameters
scDECA.run_scDECA(
    obj=adata,
    model_type="scgpt",
    embedding_key=None,
    pre_processing_flag=True,
    biogrid_flag=False,
    human_flag=False,
    number_of_batches=5,
    split_cells=False,
    n_neighbors=25,
    max_epoch=150,
    model_name="my_project",
    save_model_flag=False,
    bbknn_flag=False,
    device_str="cuda:0",
    num_heads=8,
    projection_dim=None
)
```

### **Parameters**

* **obj (AnnData, required)**:  
  AnnData object containing gene expression matrix (`.X`) and FM gene embeddings stored in `.varm`.

* **model_type (str, optional)**:  
  Type of foundation model embeddings.  
  Options: `"scgpt"`, `"cellfm"`, `"custom"`. Default: `"scgpt"`.

* **embedding_key (str, optional)**:  
  Key in `adata.varm` containing gene embeddings.  
  Automatically selected when `None`.

* **pre_processing_flag (bool, optional)**:  
  If `True`, apply Scanpy preprocessing (`log1p`, neighbors).  
  If `False`, use existing `adata.raw`. Default: `True`.

* **biogrid_flag (bool, optional)**:  
  If `True`, use BioGRID as the PPI network instead of the default STRING-like network. Default: `False`.

* **human_flag (bool, optional)**:  
  Controls gene symbol casing for PPI matching (use for human datasets). Default: `False`.

* **number_of_batches (int, optional)**:  
  Number of cell mini-batches for memory-efficient training.  
  Automatically adjusted for large datasets. Default: `5`.

* **split_cells (bool, optional)**:  
  If `True`, training is performed in *cell-batching* mode instead of full KNN-edge mode. Default: `False`.

* **n_neighbors (int, optional)**:  
  Number of neighbors used to construct the cell KNN graph. Default: `25`.

* **max_epoch (int, optional)**:  
  Maximum number of epochs for model training. Default: `150`.

* **model_name (str, optional)**:  
  Name of the folder for saving model outputs under `scDECA/Models/<model_name>`.

* **save_model_flag (bool, optional)**:  
  If `True`, save the final PyTorch model (`model.pt`). Default: `False`.

* **bbknn_flag (bool, optional)**:  
  If `True`, use BBKNN for batch-corrected neighbor graph construction. Default: `False`.

* **device_str (str, optional)**:  
  Device string such as `"cuda:0"` or `"cpu"`. Default: `"cuda:0"`.

* **num_heads (int, optional)**:  
  Number of cross-attention heads used in gene–cell alignment. Default: `8`.

* **projection_dim (int, optional)**:  
  Projection dimension for fusing FM embeddings with raw expression features. Default: `None`.

### **Loading Model-Derived Embeddings and Outputs**

```python
# Load embeddings after training
gene_embedding, cell_embedding, node_features, reconstructed_feature, normalized_feature = \
    scDECA.load_embeddings(model_name="my_project")
```

**Returns:**
* **gene_embedding (np.ndarray)**: Latent gene representations integrating FM-based global semantics with PPI-guided structure
* **cell_embedding (np.ndarray)**: Cell-level embeddings capturing context-dependent similarity patterns
* **node_features (np.ndarray)**: Z-score normalized gene expression matrix used as model input
* **reconstructed_feature (np.ndarray)**: Denoised expression matrix for downstream analysis
* **normalized_feature (np.ndarray)**: Per-batch/per-cell normalized expression values

## 📚 **Tutorials**

We provide comprehensive tutorials to guide you through the complete scDECA workflow:

### **Step 1: Foundation Model Embedding Extraction (Required First)**
**[FMs Gene Embedding Extraction](https://github.com/biomb-lab/scDECA/blob/main/tutorials/embedding_extract.ipynb)**  
This notebook demonstrates:
- Extracting scGPT gene token embeddings
- Generating CellFM embeddings (alternative)
- Storing embeddings in `adata.varm`
- Handling different species and gene naming conventions

### **Step 2: Basic scDECA Usage**
**[scDECA Basic Example](https://github.com/biomb-lab/scDECA/blob/main/tutorials/scDECA_basic.ipynb)**  
Learn the fundamentals:
- Loading scRNA-seq data with FM embeddings
- Configuring model parameters
- Training scDECA end-to-end
- Extracting and visualizing learned representations
<!-- 
### **Step 3: Advanced Applications**
**[Advanced Analysis Tutorial](https://github.com/biomb-lab/scDECA/blob/main/tutorials/scDECA_advanced.ipynb)**  
Explore advanced features:
- Batch effect correction with BBKNN
- Memory-efficient training for large datasets
- Cross-species analysis with BioGRID
- Downstream analysis: clustering, DEG, pathway enrichment

### **Step 4: Benchmarking and Evaluation**
**[Benchmark Tutorial](https://github.com/biomb-lab/scDECA/blob/main/tutorials/benchmark.ipynb)**  
Compare scDECA with other methods:
- Imputation quality metrics
- Co-expression recovery evaluation
- Gene function prediction benchmarks
- Visualization of results -->

## 📊 **Example Datasets**

We provide preprocessed example datasets with FM embeddings:

  Download: [Google Drive](https://drive.google.com/file/d/1jNmiCSoZvXwdhfdHKWQKjdkW-CgA_J9H/view?usp=share_link)


## 📑 **Citation**

Myeongbin Oh, Minsik Oh*. scDECA: Integrating Gene-Cell Interactions with Global Priors and Local Structures in Single-Cell Transcriptomics. (Manuscript in preparation)

*Corresponding author

(Official BibTeX will be added once the paper is available)

## 📧 **Contact**

For questions, issues, or collaboration inquiries:

- **Email**: [bin000815@mju.ac.kr]
- **GitHub Issues**: [https://github.com/biomb-lab/scDECA/issues](https://github.com/biomb-lab/scDECA/issues)

## 🙏 **Thank You**

This code is based on the [scNET](https://github.com/madilabcode/scNET) framework.  
We thank the scNET authors for providing their codebase, which served as the foundation for developing scDECA.

