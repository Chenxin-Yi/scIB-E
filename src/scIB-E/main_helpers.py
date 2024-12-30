from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scib_metrics.benchmark import Benchmarker

def determine_batch_correction_method(batch_constraint=None, celltype_constraint=None, batch_celltype_constraint=None):
    """
    Determine the data integration method based on the given constraints.
    
    Parameters:
    - batch_constraint: The batch constraint (can be None).
    - celltype_constraint: The cell type constraint (can be None).
    
    Returns:
    - The data integration method as a string.
    """
    # scVI
    if batch_constraint is None and celltype_constraint is None and batch_celltype_constraint is None:
        return None
    
    # level-1
    if batch_constraint is not None and celltype_constraint is None:
        return batch_constraint
    
    # level-2
    if batch_constraint is None and celltype_constraint is not None:
        return celltype_constraint
    
    # level-3
    if batch_celltype_constraint:
        return batch_celltype_constraint
    return f"{batch_constraint}_{celltype_constraint}"


def generate_save_dir(params):
    """
    Generate the save directory based on the provided parameters.

    Parameters:
        params: A dictionary containing information such as the dataset name, batch constraint, cell type constraint, etc.

    Returns:
        save_dir: The path to the save directory.
    """

    dataset_name = params.get('dataset_name')
    batch_representation = params.get('batch_representation')
    emb_merge = params.get('emb_merge')
    
    batch_constraint = params.get('batch_constraint')
    celltype_constraint = params.get('celltype_constraint')
    batch_celltype_constraint = params.get('batch_celltype_constraint')
    batch_constraint_weight = params.get('batch_constraint_weight')
    celltype_constraint_weight = params.get('celltype_constraint_weight')
    batch_celltype_constraint_weight = params.get('batch_celltype_constraint_weight')
    
    # scVI
    if not batch_constraint and not celltype_constraint and not batch_celltype_constraint:
        save_dir = Path(f"save/{dataset_name}/scvi/batch_{batch_representation}_{emb_merge}")
    
    # level-1
    elif batch_constraint and not celltype_constraint:
        corr_mse_weight = params.get('corr_mse_weight')
        save_dir = Path(f"save/{dataset_name}/level1/{batch_constraint}-w{batch_constraint_weight}/corr-mse-w{corr_mse_weight}/batch_{batch_representation}_{emb_merge}")
        
    # level-2
    elif not batch_constraint and celltype_constraint:
        corr_mse_weight = params.get('corr_mse_weight')
        save_dir = Path(f"save/{dataset_name}/level2/{celltype_constraint}-w{celltype_constraint_weight}/corr-mse-w{corr_mse_weight}/batch_{batch_representation}_{emb_merge}")

    # level-3
    else:
        if batch_celltype_constraint:
            corr_mse_weight = params.get('corr_mse_weight')
            save_dir = Path(f"save/{dataset_name}/level3/{batch_celltype_constraint}-w{batch_celltype_constraint_weight}/corr-mse-w{corr_mse_weight}/batch_{batch_representation}_{emb_merge}")

        else: 
            corr_mse_weight = params.get('corr_mse_weight')
            save_dir = Path(f"save/new/{dataset_name}/level3/{batch_constraint}-w{batch_constraint_weight}-{celltype_constraint}-w{celltype_constraint_weight}/corr-mse-w{corr_mse_weight}/batch_{batch_representation}_{emb_merge}")
 
    return save_dir

def plot_all_loss(history, save_dir):
    train_metrics = pd.DataFrame(history.get('train_loss_epoch', []))
    val_metrics = pd.DataFrame(history.get('validation_loss', []))
    epochs = range(1, len(train_metrics) + 1)
    plt.figure(figsize=(8, 8))
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation loss')
    plt.xlabel("Epochs")
    plt.ylabel("loss")
    plt.legend(['train_loss_epoch', 'validation_loss'])
    plt.savefig(save_dir / 'all_loss.png')

def plot_scvi_loss(history, metric, save_dir):
    train_metrics = pd.DataFrame(history.get(metric + "_train", []))
    val_metrics = pd.DataFrame(history.get(metric + "_validation", []))
    epochs = range(1, len(train_metrics) + 1)
    plt.figure(figsize=(8, 8))
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation ' + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_" + metric, 'val_' + metric])
    plt.savefig(save_dir / f"{metric}.png")

def plot_batch_correction_loss(history, batch_constraint, celltype_constraint, batch_celltype_constraint, save_dir):
    for key in history.keys():
        if "epoch" in key and ((batch_constraint and batch_constraint in key) or (celltype_constraint and celltype_constraint in key) or (batch_celltype_constraint and batch_celltype_constraint in key)):
            metrics = pd.DataFrame(history[key])
            epochs = range(1, len(metrics) + 1)
            plt.figure(figsize=(8, 8))
            plt.plot(epochs, metrics, 'ro-')
            plt.title(key)
            plt.xlabel("Epochs")
            plt.ylabel(key)
            plt.legend([key])
            plt.savefig(save_dir / f"{key}.png")

def plot_corr_mse_loss(history, save_dir):
    for key in history.keys():
        if "epoch" in key and "corr" in key:
            metrics = pd.DataFrame(history[key])
            epochs = range(1, len(metrics) + 1)
            plt.figure(figsize=(8, 8))
            plt.plot(epochs, metrics, 'ro-')
            plt.title(key)
            plt.xlabel("Epochs")
            plt.ylabel(key)
            plt.legend([key])
            plt.savefig(save_dir / f"{key}.png")

def plot_loss(history, save_dir, batch_constraint=None, celltype_constraint=None, batch_celltype_constraint=None):
    """
    Plot and save various training and validation metrics.

    Parameters:
        history: Dictionary containing training and validation metrics.
        batch_constraint: Batch constraint method used.
        celltype_constraint: Cell type constraint method used.
        save_dir: Directory to save plots.
    """
    plot_all_loss(history, save_dir)
    plot_scvi_loss(history, "kl_local", save_dir)
    plot_scvi_loss(history, "reconstruction_loss", save_dir)
    if batch_constraint or celltype_constraint or batch_celltype_constraint:
        plot_batch_correction_loss(history, batch_constraint, celltype_constraint, batch_celltype_constraint, save_dir)
        plot_corr_mse_loss(history, save_dir)

def plot_scanvi_loss(history, metric, save_dir):
    train_metrics = pd.DataFrame(history.get("train_"+ metric, []))
    val_metrics = pd.DataFrame(history.get("validation_" + metric, []))
    epochs = range(1, len(train_metrics) + 1)
    plt.figure(figsize=(8, 8))
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation ' + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_" + metric, 'val_' + metric])
    plt.savefig(save_dir / f"{metric}.png")

def plot_loss_scanvi(history, save_dir):
    """
    Plot and save various training and validation metrics.

    Parameters:
        history: Dictionary containing training and validation metrics.
        batch_constraint: Batch constraint method used.
        celltype_constraint: Cell type constraint method used.
        save_dir: Directory to save plots.
    """
    plot_all_loss(history, save_dir)
    plot_scvi_loss(history, "kl_local", save_dir)
    plot_scvi_loss(history, "reconstruction_loss", save_dir)
    plot_scanvi_loss(history, "classification_loss", save_dir) 

def calculate_mean_ja(adata, batch_key, pca_key='X_pca_batch', latent_key="X_scVI", k=15):
    unique_batches = np.unique(adata.obs[batch_key])
    all_ja = []

    for batch in unique_batches:

        batch_mask = adata.obs[batch_key] == batch
        X_pca_batch = adata.obsm[pca_key][batch_mask]
        latent_batch = adata.obsm[latent_key][batch_mask]

        nbrs1 = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(X_pca_batch)
        A1 = nbrs1.kneighbors_graph(X_pca_batch, k, mode='connectivity')

        nbrs2 = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(latent_batch)
        A2 = nbrs2.kneighbors_graph(latent_batch, k, mode='connectivity')

        A1.setdiag(0)
        A2.setdiag(0)

        A1 = A1.maximum(A1.T)
        A2 = A2.maximum(A2.T)

        intersection = A1.multiply(A2)
        union = A1.maximum(A2)

        # Calculate Jaccard Index
        jaccard_index = (intersection.nnz) / (union.nnz)
        all_ja.append(jaccard_index)

    total_jaccard_index = np.mean(all_ja)

    return total_jaccard_index

def calculate_scIB_E_results(adata, params, SCVI_LATENT_KEY):
    """
    Calculate scIB-E benchmark results.

    Parameters:
    -----------
    adata : AnnData
        The annotated data object.
    params : dict
        A dictionary containing batch key and labels key.
    SCVI_LATENT_KEY : str
        The key for the latent representation in `obsm`.

    Returns:
    --------
    scIB_E_results : DataFrame
        The scIB-E benchmark results.
    """
    bm = Benchmarker(
        adata,
        batch_key=params["batch_key"],
        label_key=params["labels_key"],
        embedding_obsm_keys=[SCVI_LATENT_KEY],
        n_jobs=1,
    )
    bm.benchmark()
    benchmark_results = bm.get_results(min_max_scale=False)

    scIB_E_results = benchmark_results.iloc[:, :-3]
    scIB_E_results["Jaccard index"] = calculate_mean_ja(adata, params["batch_key"], 'X_pca_batch', SCVI_LATENT_KEY)
    scIB_E_results["Jaccard index"] = scIB_E_results["Jaccard index"].astype(object)

    metrics = {
        "Batch correction": ["Silhouette batch", "iLISI", "KBET", "Graph connectivity"],
        "Inter cell-type conservation": ["Isolated labels", "KMeans NMI", "KMeans ARI", "Silhouette label", "cLISI"],
        "Intra cell-type conservation": ["PCR comparison", "Jaccard index"]
    }

    for metric, columns in metrics.items():
        scIB_E_results.loc[SCVI_LATENT_KEY, metric] = scIB_E_results.loc[SCVI_LATENT_KEY, columns].mean()

    scIB_E_total_score = (
        0.2 * scIB_E_results.loc[SCVI_LATENT_KEY, "Batch correction"]
        + 0.4 * scIB_E_results.loc[SCVI_LATENT_KEY, "Inter cell-type conservation"]
        + 0.4 * scIB_E_results.loc[SCVI_LATENT_KEY, "Intra cell-type conservation"]
    )
    scIB_E_results.loc[SCVI_LATENT_KEY, "scIB-E Total score"] = scIB_E_total_score

    score_columns = ["Batch correction", "Inter cell-type conservation", "Intra cell-type conservation", "scIB-E Total score"]
    scIB_E_results[score_columns] = scIB_E_results[score_columns].astype(object)

    metric_type_mapping = metrics.copy()
    metric_type_mapping["Aggregate score"] = [
        "Batch correction", "Inter cell-type conservation", "Intra cell-type conservation", "scIB-E Total score"
    ]

    for metric_type, columns in metric_type_mapping.items():
        scIB_E_results.loc["Metric Type", columns] = metric_type

    return scIB_E_results
