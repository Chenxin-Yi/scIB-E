import os
import scanpy as sc
import seaborn as sns
import torch
import pandas as pd
from main_helpers import determine_batch_correction_method, generate_save_dir, plot_loss, calculate_scIB_E_results

# setup environment
os.chdir("/home/Project/scIB-E") # change working directory to project root
os.environ['CUDA_VISIBLE_DEVICES']='0'  # set GPU device
sc.set_figure_params(figsize=(4, 4), frameon=False)
sns.set_theme()
torch.set_float32_matmul_precision("high")

import scvi
scvi.settings.seed = 0

# hyperparameters
params = dict(
    # dataset
    dataset_name = "immune_processed",
    batch_key="batch",
    labels_key="final_annotation",

    # batch_constraint: Defines the constraint for batch effect.
    #   - None: No batch constraint applied.
    #   - Specific method (e.g., "RCE"): Applied for batch removal.
    batch_constraint = None,
    batch_constraint_weight = 0,

    # celltype_constraint: Defines the constraint for cell-type information.
    #   - None: No cell-type constraint applied.
    #   - Specific method (e.g., "CellSupcon"): Applied for cell-type incorporation.
    celltype_constraint = "CellSupcon",
    celltype_constraint_weight = 10,

    # batch_celltype_constraint: Defines the combined constraint for both batch and cell-type.
    #   - None: No batch-cell-type constraint applied.
    #   - "DomainClassTripletLoss": Applies the specific method for both batch and cell-type constraints.
    #   - For other methods, both batch_constraint and celltype_constraint must be specified.

    # batch_celltype_constraint = "DomainClassTripletLoss",
    # batch_celltype_constraint_weight = batch_celltype_constraint_weight,

    corr_mse_weight = 10,  # weight for Corr-MSE loss

    # model params
    n_layers = 2,
    n_latent = 30,
    gene_likelihood = "zinb",
    batch_representation = "embedding",      # default: "one-hot"
    batch_embedding_kwargs = {"embedding_dim": 30},
    emb_merge = "cat",      # ["add", "cat"] default: "cat"

    # train params
    max_epochs = 400, # 400     
    train_size = 0.9,   
    shuffle_set_split = True,
    batch_size = 128,
    early_stopping = True,  # default: False

    # plot loss
    plot_loss = False,
)

batch_correction_method = determine_batch_correction_method(params["batch_constraint"], params["celltype_constraint"])

# batch_correction_method = determine_batch_correction_method(batch_celltype_constraint=params["batch_celltype_constraint"])  # DomainClassTripletLoss

model_params = dict(
    n_layers=params["n_layers"],
    n_latent=params["n_latent"],
    gene_likelihood=params["gene_likelihood"],
    batch_representation=params["batch_representation"],
    batch_embedding_kwargs=params["batch_embedding_kwargs"],
    emb_merge=params["emb_merge"],
)

train_params = dict(
    max_epochs=params["max_epochs"],    
    train_size=params["train_size"],
    shuffle_set_split=params["shuffle_set_split"],
    batch_size=params["batch_size"],
    early_stopping=params["early_stopping"],
    batch_correction_method=batch_correction_method,
    batch_correction_weight={"batch_constraint_weight": params["batch_constraint_weight"],
                             "celltype_constraint_weight":params["celltype_constraint_weight"],
                             "corr_mse_weight":params["corr_mse_weight"]},
    # batch_correction_weight={"batch_celltype_constraint_weight": params["batch_celltype_constraint_weight"],
    #                          "corr_mse_weight":corr_mse_weight},  # DomainClassTripletLoss
)

# load data
adata = sc.read(f"data/{params['dataset_name']}.h5ad")
scvi.model.SCVI.setup_anndata(adata, layer="counts", pca_key = "X_pca_batch", batch_key=params["batch_key"], labels_key=params["labels_key"])

# train model
model = scvi.model.SCVI(adata, **model_params)
model.train(**train_params)
history = model.history

# save results
save_dir = generate_save_dir(params)
save_dir.mkdir(parents=True, exist_ok=True)

if params["plot_loss"]:
    plot_loss(history, save_dir, batch_constraint=params["batch_constraint"], celltype_constraint=params["celltype_constraint"])
    # plot_loss(history, save_dir, batch_celltype_constraint=params["batch_celltype_constraint"])  # DomainClassTripletLoss

SCVI_LATENT_KEY = "X_scVI"
adata.obsm[SCVI_LATENT_KEY] = model.get_latent_representation()

sc.pp.neighbors(adata, use_rep=SCVI_LATENT_KEY)
sc.tl.umap(adata, min_dist=0.3)
fig = sc.pl.umap(
    adata,
    color=[params["batch_key"], params["labels_key"]],
    title=[params["batch_key"], params["labels_key"]],
    frameon=False,
    return_fig=True,
    show=False,
    wspace=0.4,
    palette=sc.pl.palettes.vega_20_scanpy
)
fig.savefig(save_dir / "umap_scVI.png", dpi=300) 

latent_df = pd.DataFrame(adata.obsm[SCVI_LATENT_KEY])
latent_df.to_csv(save_dir / "X_scVI.csv", index=False, header=False)

scIB_E_results = calculate_scIB_E_results(adata, params, SCVI_LATENT_KEY)
scIB_E_results.to_csv(save_dir / "scIB_E_results.csv", index=False)