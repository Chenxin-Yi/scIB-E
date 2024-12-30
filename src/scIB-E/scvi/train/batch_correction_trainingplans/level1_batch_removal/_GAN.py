import torch
from scvi.train._trainingplans import *

# The GAN component is adapted from
# Title: Deep feature extraction of single-cell transcriptomes by generative adversarial network
# Authors: Mojtaba Bahrami, et al.
# Code: https://github.com/li-lab-mcgill/singlecell-deepfeature/blob/master/scripts/utils_helper.py

class GAN_TrainingPlan(TrainingPlan):
    """Training plan for VAEs with Generative Adversarial Network (GAN) loss to encourage latent space mixing.

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
    adversarial_classifier
        Whether to use adversarial classifier in the latent space
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
        adversarial_classifier: Union[bool, Classifier] = True,
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
        if adversarial_classifier is True:
            self.n_output_classifier = self.module.n_batch
            self.adversarial_classifier = Classifier(
                n_input=self.module.n_latent,
                n_hidden=32,
                n_labels=self.n_output_classifier,
                n_layers=2,
                logits=True,
            )
        else:
            self.adversarial_classifier = adversarial_classifier
        self.automatic_optimization = False

        self.batch_correction_weight = batch_correction_weight["batch_constraint_weight"]
        self.corr_mse_weight = batch_correction_weight["corr_mse_weight"]
        self.mse_loss = torch.nn.MSELoss()

    def loss_adversarial_classifier(self, z, batch_index, predict_true_class=True):
        """Loss for adversarial classifier."""
        torch.manual_seed(0)
        n_classes = self.n_output_classifier
        cls_logits = torch.nn.LogSoftmax(dim=1)(self.adversarial_classifier(z))

        if predict_true_class:
            cls_target = one_hot(batch_index, n_classes)
        else:
            one_hot_batch = one_hot(batch_index, n_classes)
            # place zeroes where true label is
            cls_target = (~one_hot_batch.bool()).float()
            cls_target = cls_target / (n_classes - 1)

        l_soft = cls_logits * cls_target
        loss = -l_soft.sum(dim=1).mean()

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
        """Training step for adversarial training."""
        if "kl_weight" in self.loss_kwargs:
            self.loss_kwargs.update({"kl_weight": self.kl_weight})

        batch_label = batch[REGISTRY_KEYS.BATCH_KEY]
        X_pca = batch[REGISTRY_KEYS.PCA_KEY]
        opts = self.optimizers()
        opt1, opt2, opt3 = opts

        inference_outputs, _, scvi_loss = self.forward(batch, loss_kwargs=self.loss_kwargs)
        z = inference_outputs["z"]
        loss = scvi_loss.loss

        total_corr_mse_loss = 0.0
        unique_batch_labels = torch.unique(batch_label)
        for lbl in unique_batch_labels:
            mask = batch_label == lbl
            mask = mask.squeeze()
            X_pca_batch = X_pca[mask]
            z_batch = z[mask]
            corr_mse_loss = self.corr_mse_weight * self.calculate_corr_mse(X_pca_batch, z_batch)
            total_corr_mse_loss += corr_mse_loss

        loss += total_corr_mse_loss

        self.log("corr_mse_loss", total_corr_mse_loss, on_epoch=True, prog_bar=True)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        
        self.compute_and_log_metrics(scvi_loss, self.train_metrics, "train")
        opt1.zero_grad()
        self.manual_backward(loss,retain_graph=True)
        opt1.step()

        inference_outputs, _, _ = self.forward(batch, loss_kwargs=self.loss_kwargs)
        z = inference_outputs["z"]

        # Update D Net
        for disc_iter in range(10):
            Dloss = self.loss_adversarial_classifier(z.detach(), batch_label, True)
            self.log("GAN_Dloss", Dloss, on_epoch=True, prog_bar=True)
            opt2.zero_grad()
            self.manual_backward(Dloss,retain_graph=True)
            opt2.step()

        # Update E Net
        Eloss = -self.batch_correction_weight * self.loss_adversarial_classifier(z, batch_label, True)
        self.log("GAN_Eloss", Eloss, on_epoch=True, prog_bar=True)
        opt3.zero_grad()
        self.manual_backward(Eloss)
        opt3.step()

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

        if self.adversarial_classifier is not False:
            params2 = filter(lambda p: p.requires_grad, self.adversarial_classifier.parameters())
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
        return config1