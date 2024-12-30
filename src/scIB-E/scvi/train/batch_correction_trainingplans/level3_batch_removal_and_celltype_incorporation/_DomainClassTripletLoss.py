import torch
from torch.nn import functional as F
from scvi.train._trainingplans import *
from torch import nn

# The domain class triplet loss component is adapted from
# Title: Domain-aware triplet loss in domain generalization
# Authors: Kaiyu Guo, Brian C. Lovell
# Code: https://github.com/workerbcd/DCT/blob/main/domainbed/losses/tripletloss.py

class DomainClassTripletLoss_TrainingPlan(TrainingPlan):
    """Training plan for VAEs with Domain class triplet loss to encourage latent space mixing.

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

        self.batch_correction_weight = batch_correction_weight["batch_celltype_constraint_weight"]
        self.hard_factor = 0
        self.margin = None
        self.leri = False
        if self.margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=self.margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()
        self.mse_loss = torch.nn.MSELoss()
        self.corr_mse_weight = batch_correction_weight["corr_mse_weight"]

    def euclidean_dist(self, x, y):
        """
        Args:
        x: pytorch Variable, with shape [m, d]
        y: pytorch Variable, with shape [n, d]
        Returns:
        dist: pytorch Variable, with shape [m, n]
        """
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist = dist - 2 * torch.matmul(x, y.t())
        # dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist
    
    def domain_hard_sample_mining(self, dist_mat, labels, domains, leri=None):
        assert len(dist_mat.size()) == 2
        assert dist_mat.size(0) == dist_mat.size(1)
        N = dist_mat.size(0)

        is_domain_pos = domains.expand(N,N).eq(domains.expand(N, N).t())
        is_domain_neg = domains.expand(N,N).ne(domains.expand(N,N).t())
        is_2 = domains.expand(N,N).eq(torch.ones(N,N).cuda())
        is_label_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
        is_label_neg = labels.expand(N, N).ne(labels.expand(N, N).t())
        domainpos_labelneg = is_label_neg & is_domain_pos
        domainneg_labelpos = is_label_pos & is_domain_neg

        if leri:
            is_domain_pos = is_domain_pos | is_2
            is_domain_neg = is_domain_neg | is_2
            domainpos_labelneg = is_label_neg & is_domain_pos
            domainneg_labelpos = is_label_pos & is_domain_neg


        dist_dpln,dist_dnlp =[],[]
        for i in range(N):
            if dist_mat[i][domainneg_labelpos[i]].shape[0]!=0:
                dist_dnlp.append(torch.max(dist_mat[i][domainneg_labelpos[i]].contiguous(), 0, keepdim=True)[0])
            else:
                dist_dnlp.append(torch.zeros(1).cuda())
            if dist_mat[i][domainpos_labelneg[i]].shape[0]!=0:
                dist_dpln.append(torch.min(dist_mat[i][domainpos_labelneg[i]].contiguous(), 0, keepdim=True)[0])
            else:
                dist_dpln.append(torch.zeros(1).cuda())

        dist_dnlp = torch.cat(dist_dnlp).clamp(min=1e-12)
        dist_dpln = torch.cat(dist_dpln).clamp(min=1e-12)

        return dist_dnlp,dist_dpln

    def domain_triplet_loss(self, global_feat, labels, batch_labels):
        dist_mat = self.euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = self.domain_hard_sample_mining(dist_mat, labels, batch_labels, leri=self.leri)

        dist_ap *= (1.0 + self.hard_factor)
        dist_an *= (1.0 - self.hard_factor)

        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None :
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
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
        opt1 = self.optimizers()
        inference_outputs, _, scvi_loss = self.forward(batch, loss_kwargs=self.loss_kwargs)

        loss = scvi_loss.loss
        z = inference_outputs["z"]
        dct_loss = self.batch_correction_weight * self.domain_triplet_loss(z, celltype_labels.squeeze(1), batch_tensor.squeeze(1))
        loss += dct_loss

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
        self.log("DomainClassTripletLoss_loss", dct_loss, on_epoch=True, prog_bar=True)
        self.compute_and_log_metrics(scvi_loss, self.train_metrics, "train")
        opt1.zero_grad()
        self.manual_backward(loss,retain_graph=True)
        opt1.step()

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
        dct_loss = self.batch_correction_weight * self.domain_triplet_loss(z, celltype_labels.squeeze(1), batch_tensor.squeeze(1))
        loss += dct_loss

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
        parameters_to_optimize = list(self.module.parameters())
        params1 = filter(lambda p: p.requires_grad, parameters_to_optimize)
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

        return config1