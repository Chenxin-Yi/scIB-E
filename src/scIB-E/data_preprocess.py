import scanpy as sc
import os
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
os.chdir("/home/Project/scIB-E") # change working directory to project root

adata = sc.read('data/raw_data/Immune_ALL_human.h5ad')  # load the downloaded data stored in the 'raw_data' directory

sc.pp.highly_variable_genes(
    adata,
    flavor="seurat_v3",
    n_top_genes=4000,
    layer="counts",
    batch_key="batch",
    subset=True,
)

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Perform PCA for each batch and store the results
X_pca_batch = np.zeros((adata.shape[0], 50))  # 50 principal components
batches = adata.obs["batch"].unique()
for batch in batches:
    adata_batch = adata[adata.obs["batch"] == batch].copy()
    sc.pp.pca(adata_batch, n_comps=50, svd_solver='arpack')
    X_pca_batch[adata.obs["batch"] == batch] = adata_batch.obsm['X_pca'] 
adata.obsm['X_pca_batch'] = X_pca_batch

sc.write("data/immune_processed.h5ad", adata)