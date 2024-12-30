import torch
import torch.nn as nn
from torch.nn import functional as F
from scvi.train._trainingplans import *

# The MIM component is adapted from
# # Title: Club: A contrastive log-ratio upper bound of mutual information
# Authors: Pengyu Cheng, et al.
# Code: https://github.com/Linear95/CLUB/blob/master/mi_estimators.py

# The CellSupcon component is adapted from
# Title: Supervised contrastive learning
# Authors: Prannay Khosla, et al.
# Code: https://github.com/HobbitLong/SupContrast/blob/master/losses.py

class CLUBSample(nn.Module):  # Sampled version of the CLUB estimator
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUBSample, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim),
                                       nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
        
    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        
        sample_size = x_samples.shape[0]
        #random_index = torch.randint(sample_size, (sample_size,)).long()
        random_index = torch.randperm(sample_size).long()
        
        positive = - (mu - y_samples)**2 / logvar.exp()
        negative = - (mu - y_samples[random_index])**2 / logvar.exp()
        upper_bound = (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()
        return upper_bound/2.

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)

class MIM_CellSupcon_TrainingPlan(TrainingPlan):
    """Training plan for VAEs with MIM loss and CellSupcon loss to encourage latent space mixing.

    Parameters
    ----------
    module
        A module instance from class ``BaseModuleClass``.
    optimizer
        One of "Adam" (:class:`~torch.optim.Adam`), "AdamW" (:class:`~torch.optim.AdamW`),
        or "Custom", which requires a custom optimizer creator callable to be passed via
        `optimizer_creator`.
    optimizer_creator
        A callable taking in parameters and returning a :class:`~torch.optim.Optimizer`.
        This allows using any PyTorch optimizer with custom hyperparameters.
    lr
        Learning rate used for optimization, when `optimizer_creator` is None.
    weight_decay
        Weight decay used in optimization, when `optimizer_creator` is None.
    eps
        eps used for optimization, when `optimizer_creator` is None.
    n_steps_kl_warmup
        Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
        Only activated when `n_epochs_kl_warmup` is set to None.
    n_epochs_kl_warmup
        Number of epochs to scale weight on KL divergences from 0 to 1.
        Overrides `n_steps_kl_warmup` when both are not `None`.
    reduce_lr_on_plateau
        Whether to monitor validation loss and reduce learning rate when validation set
        `lr_scheduler_metric` plateaus.
    lr_factor
        Factor to reduce learning rate.
    lr_patience
        Number of epochs with no improvement after which learning rate will be reduced.
    lr_threshold
        Threshold for measuring the new optimum.
    lr_scheduler_metric
        Which metric to track for learning rate reduction.
    lr_min
        Minimum learning rate allowed
    batch_correction_weight
        Weights used for batch correction in the training process.
    **loss_kwargs
        Keyword args to pass to the loss method of the `module`.
        `kl_weight` should not be passed here and is handled automatically.
    """

    def __init__(
        self,
        module: BaseModuleClass,
        *,
        optimizer: Literal["Adam", "AdamW", "Custom"] = "Adam",
        optimizer_creator: Optional[TorchOptimizerCreator] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        n_steps_kl_warmup: int = None,
        n_epochs_kl_warmup: int = 400,
        reduce_lr_on_plateau: bool = False,
        lr_factor: float = 0.6,
        lr_patience: int = 30,
        lr_threshold: float = 0.0,
        lr_scheduler_metric: Literal[
            "elbo_validation", "reconstruction_loss_validation", "kl_local_validation"
        ] = "elbo_validation",
        lr_min: float = 0,
        batch_correction_weight: dict = None,
        **loss_kwargs,
    ):
        super().__init__(
            module=module,
            optimizer=optimizer,
            optimizer_creator=optimizer_creator,
            lr=lr,
            weight_decay=weight_decay,
            n_steps_kl_warmup=n_steps_kl_warmup,
            n_epochs_kl_warmup=n_epochs_kl_warmup,
            reduce_lr_on_plateau=reduce_lr_on_plateau,
            lr_factor=lr_factor,
            lr_patience=lr_patience,
            lr_threshold=lr_threshold,
            lr_scheduler_metric=lr_scheduler_metric,
            lr_min=lr_min,
            **loss_kwargs,
        )
        self.automatic_optimization = False

        self.club_dim = self.module.z_encoder.mean_encoder.out_features
        self.mi_net = CLUBSample(self.club_dim, self.club_dim, 512)

        self.temperature=0.07
        self.contrast_mode='one'
        self.base_temperature=0.07

        self.batch_constraint_weight = batch_correction_weight["batch_constraint_weight"]
        self.celltype_constraint_weight = batch_correction_weight["celltype_constraint_weight"]
        
        self.mse_loss = torch.nn.MSELoss()
        self.corr_mse_weight = batch_correction_weight["corr_mse_weight"]
    
    
    def supervised_contrastive_loss(self, features, labels=None, mask=None):
        """Compute Supervised Contrastive Loss."""
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        device = features.device

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = 1
        contrast_feature = features
        if self.contrast_mode == 'one':
            anchor_feature = features
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    
    def calculate_corr_mse(self, matrix1, matrix2, mask=None):
        corr_matrix1 = torch.corrcoef(matrix1)
        corr_matrix2 = torch.corrcoef(matrix2)
        if mask is not None:
            corr_matrix1 = corr_matrix1[mask]
            corr_matrix2 = corr_matrix2[mask]
        corr_mse_loss = self.mse_loss(corr_matrix2, corr_matrix1)
        return corr_mse_loss

    def training_step(self, batch, batch_idx):
        """Training step for Supervised_contrastive training."""
        if "kl_weight" in self.loss_kwargs:
            self.loss_kwargs.update({"kl_weight": self.kl_weight})

        batch_tensor = batch[REGISTRY_KEYS.BATCH_KEY]
        celltype_labels = batch[REGISTRY_KEYS.LABELS_KEY]
        X_pca = batch[REGISTRY_KEYS.PCA_KEY]

        opts = self.optimizers()
        opt1, opt2, opt3 = opts

        inference_outputs, _, scvi_loss = self.forward(batch, loss_kwargs=self.loss_kwargs)
        loss = scvi_loss.loss

        z = inference_outputs["z"]
        supcon_loss = self.celltype_constraint_weight * self.supervised_contrastive_loss(z, celltype_labels)
        loss += supcon_loss

        total_corr_mse_loss = 0.0
        unique_batch_labels = torch.unique(batch_tensor)

        for lbl in unique_batch_labels:
            mask = batch_tensor == lbl
            mask = mask.squeeze()
            X_pca_batch = X_pca[mask]
            z_batch = z[mask]
            corr_mse_loss = self.corr_mse_weight * self.calculate_corr_mse(X_pca_batch, z_batch)
            total_corr_mse_loss += corr_mse_loss

        loss += total_corr_mse_loss

        self.log("corr_mse_loss", total_corr_mse_loss, on_epoch=True, prog_bar=True)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("CellSupcon_loss", supcon_loss, on_epoch=True, prog_bar=True)
        self.compute_and_log_metrics(scvi_loss, self.train_metrics, "train")
        opt1.zero_grad()
        self.manual_backward(loss,retain_graph=True)
        opt1.step()

        inference_outputs, _, _ = self.forward(batch, loss_kwargs=self.loss_kwargs)
        z = inference_outputs["z"]
        batch_embs = self.module._embeddings_dict.batch(torch.squeeze(batch_tensor))

        # Update mi_net
        for disc_iter in range(10):
            lld_loss = 10*self.mi_net.learning_loss(z.detach(), batch_embs.detach())
            self.log("lld_loss", lld_loss, on_epoch=True, prog_bar=True)
            opt2.zero_grad()
            self.manual_backward(lld_loss,retain_graph=True)
            opt2.step()

        # Update E Net
        mi_loss = self.batch_constraint_weight * self.mi_net.forward(z, batch_embs)
        self.log("MIM_loss", mi_loss, on_epoch=True, prog_bar=True)
        opt3.zero_grad()
        self.manual_backward(mi_loss)
        opt3.step()

    def validation_step(self, batch, batch_idx):
        """Validation step for the model."""
        # loss kwargs here contains `n_obs` equal to n_training_obs
        # so when relevant, the actual loss value is rescaled to number
        # of training examples
        batch_tensor = batch[REGISTRY_KEYS.BATCH_KEY]
        celltype_labels = batch[REGISTRY_KEYS.LABELS_KEY]
        X_pca = batch[REGISTRY_KEYS.PCA_KEY]
        
        inference_outputs, _, scvi_loss = self.forward(batch, loss_kwargs=self.loss_kwargs)
        loss = scvi_loss.loss

        z = inference_outputs["z"]
        batch_embs = self.module._embeddings_dict.batch(torch.squeeze(batch_tensor))
        mi_loss = self.batch_constraint_weight * self.mi_net.forward(z, batch_embs)
        loss += mi_loss

        supcon_loss = self.celltype_constraint_weight * self.supervised_contrastive_loss(z, celltype_labels)
        loss += supcon_loss

        corr_mse_loss = self.corr_mse_weight * self.calculate_corr_mse(X_pca, z)
        loss += corr_mse_loss

        self.log(
            "validation_loss",
            loss,
            on_epoch=True,
            sync_dist=self.use_sync_dist,
        )
        self.compute_and_log_metrics(scvi_loss, self.val_metrics, "validation")

    def on_train_epoch_end(self):
        """Update the learning rate via scheduler steps."""
        if "validation" in self.lr_scheduler_metric or not self.reduce_lr_on_plateau:
            return
        else:
            sch = self.lr_schedulers()
            sch.step(self.trainer.callback_metrics[self.lr_scheduler_metric])

    def on_validation_epoch_end(self) -> None:
        """Update the learning rate via scheduler steps."""
        if not self.reduce_lr_on_plateau or "validation" not in self.lr_scheduler_metric:
            return
        else:
            sch = self.lr_schedulers()
            sch.step(self.trainer.callback_metrics[self.lr_scheduler_metric])

    def configure_optimizers(self):
        """Configure optimizers for adversarial training."""
        params1 = filter(lambda p: p.requires_grad, self.module.parameters())
        optimizer1 = self.get_optimizer_creator()(params1)
        config1 = {"optimizer": optimizer1}
        if self.reduce_lr_on_plateau:
            scheduler1 = ReduceLROnPlateau(
                optimizer1,
                patience=self.lr_patience,
                factor=self.lr_factor,
                threshold=self.lr_threshold,
                min_lr=self.lr_min,
                threshold_mode="abs",
                verbose=True,
            )
            config1.update(
                {
                    "lr_scheduler": {
                        "scheduler": scheduler1,
                        "monitor": self.lr_scheduler_metric,
                    },
                },
            )
        
        params2 = filter(lambda p: p.requires_grad, self.mi_net.parameters())
        optimizer2 = torch.optim.Adam(
            params2, lr=1e-3, eps=0.01, weight_decay=self.weight_decay
        )
        config2 = {"optimizer": optimizer2}

        params3 = filter(lambda p: p.requires_grad, self.module.z_encoder.parameters()) 
        optimizer3 = torch.optim.Adam(params3, lr=1e-3, eps=0.01, weight_decay=self.weight_decay)
        config3 = {"optimizer": optimizer3}

        opts = [config1.pop("optimizer"), config2["optimizer"], config3["optimizer"]]
        if "lr_scheduler" in config1:
            scheds = [config1["lr_scheduler"]]
            return opts, scheds
        else:
            return opts