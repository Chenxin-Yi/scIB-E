import torch
from torch.nn import functional as F
from scvi.train._trainingplans import *
from torch import nn

class RCE_CE_TrainingPlan(TrainingPlan):
    """Training plan for VAEs with RCE loss and CE loss to encourage latent space mixing.

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

        self.batch_constraint_weight = batch_correction_weight["batch_constraint_weight"]
        self.celltype_constraint_weight = batch_correction_weight["celltype_constraint_weight"]

        self.n_batch = self.module.n_batch
        self.n_celltype = self.module.n_labels
        self.batch_classifier = Classifier(
                n_input=self.module.n_latent,
                n_hidden=32,
                n_labels=self.n_batch,
                n_layers=2,
                logits=True,
            )
        self.celltype_classifier = Classifier(
                n_input=self.module.n_latent,
                n_hidden=32,
                n_labels=self.n_celltype,
                n_layers=2,
                logits=True,
            )
        self.criterion_ce = nn.CrossEntropyLoss()

        self.mse_loss = torch.nn.MSELoss()
        self.corr_mse_weight = batch_correction_weight["corr_mse_weight"]

    def CE_loss(self, z, label):
        """Loss for adversarial classifier."""
        cls_logits = torch.nn.LogSoftmax(dim=1)(self.celltype_classifier(z))
        label = label.squeeze()  
        loss = self.criterion_ce(cls_logits,label)
        return loss
    
    def RCE_loss(self, z, label):
        """Loss for adversarial classifier."""
        n_classes = self.n_batch
        cls_logits = torch.nn.LogSoftmax(dim=1)(self.batch_classifier(z))
        one_hot_batch = one_hot(label, n_classes)
        cls_target = (~one_hot_batch.bool()).float()
        cls_target = cls_target / (n_classes - 1)
        loss = self.criterion_ce(cls_logits,cls_target)
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
        rce_loss = self.batch_constraint_weight * self.RCE_loss(z, batch_tensor)
        loss += rce_loss
        ce_loss = self.celltype_constraint_weight * self.CE_loss(z, celltype_labels)
        loss += ce_loss

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
        self.log("CE_loss", ce_loss, on_epoch=True, prog_bar=True)
        self.log("RCE_loss", rce_loss, on_epoch=True, prog_bar=True)
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
        rce_loss = self.batch_constraint_weight * self.RCE_loss(z, batch_tensor)
        loss += rce_loss
        ce_loss = self.celltype_constraint_weight * self.CE_loss(z, celltype_labels)
        loss += ce_loss

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
        parameters_to_optimize = list(self.module.parameters()) + list(self.batch_classifier.parameters()) + list(self.celltype_classifier.parameters())
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