import torch
from torch.nn import functional as F
from scvi.train._trainingplans import *
from torch import nn, autograd
from torch.autograd import Variable

# The domain meta-learning component is adapted from
# Title: Domain generalization via semi-supervised meta learning
# Authors: Hossein Sharifi-Noghabi, et al.
# Code: https://github.com/hosseinshn/DGSML/blob/master/dgsml-gpu/DGMSL_fncGPU.py

class Domain_meta_learning_TrainingPlan(TrainingPlan):
    """Training plan for VAEs with Domain meta-learning loss to encourage latent space mixing.

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
        self.celltype_constraint_weight = batch_correction_weight["celltype_constraint_weight"]
        self.n_celltype = self.module.n_labels
        self.celltype_classifier = Classifier(
                n_input=self.module.n_latent,
                n_hidden=32,
                n_labels=self.n_celltype,
                n_layers=2,
                logits=True,
            )
        self.criterion_ce = nn.CrossEntropyLoss()
        self.domain = self.module.n_batch
        self.meta_train_domain = 3  # 3
        self.meta_lr = 1e-4
        self.global_coef = 1e-2 
        self.mse_loss = torch.nn.MSELoss()
        self.corr_mse_weight = batch_correction_weight["corr_mse_weight"]

    def class_centroid(self, F, Y):
        number_class = self.n_celltype
        res = torch.zeros([number_class, F.size(1)])
        ls_class = torch.unique(Y)
        ls_class = torch.sort(ls_class)[0]
        for ite in ls_class:
            res[ite,:] = F[Y==ite].mean(0)
        return res.cuda()
    
    def cloned_state_dict(self, Model):
        cloned_state_dict = {
            key: val.clone()
            for key, val in Model.state_dict().items()
        }
        return cloned_state_dict  

    def row_pairwise_distances(self, X, y=None, dist_mat=None):
        f_mat = []
        for x in X:
            y = None
            dist_mat = None
            if y is None:
                y = x
            if dist_mat is None:
                dtype = x.data.type()
                dist_mat = Variable(torch.Tensor(x.size()[0], y.size()[0]).type(dtype))

            for i, row in enumerate(x.split(1)):
                r_v = row.expand_as(y)
                sq_dist = torch.sum((r_v - y) ** 2, 1)
                dist_mat[i] = sq_dist.view(1, -1)
            f_mat.append(dist_mat)   
        return f_mat

    def global_alignment(self, mat1, mat2):
        dist_norm = 0
        for i in range(len(mat1)):
            for j in range(len(mat2)):
                dist_norm += torch.norm(F.normalize(mat1[i])-F.normalize(mat2[j])) 
        return dist_norm
    
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
        batch_label = batch[REGISTRY_KEYS.BATCH_KEY]
        X_pca = batch[REGISTRY_KEYS.PCA_KEY]

        opts = self.optimizers()
        opt1, opt2 = opts

        inference_outputs, _, scvi_loss = self.forward(batch, loss_kwargs=self.loss_kwargs)
        loss = scvi_loss.loss

        z = inference_outputs["z"]
        cls_logits = torch.nn.LogSoftmax(dim=1)(self.celltype_classifier(z))

        index = np.random.permutation(self.domain)
        meta_train_idx = index[0:self.meta_train_domain]
        meta_test_idx = index[self.meta_train_domain:]

        meta_train_loss = 0
        meta_test_loss = 0
        centroids_tr = []
        dist_mat_tr = []
        dist_mat_val = [] 

        Loss_test = 0   
        centroids_val = []

        for i in meta_train_idx:
            mask = (batch[REGISTRY_KEYS.BATCH_KEY] == i)
            if torch.all(~mask):
                continue
            z_meta_train = z[mask.squeeze(1)]
            cls_logits_meta_train = cls_logits[mask.squeeze(1)]
            celltype_meta_train = batch[REGISTRY_KEYS.LABELS_KEY][mask]

            ce_loss = self.criterion_ce(cls_logits_meta_train, celltype_meta_train)

            ctr_meta_train = self.class_centroid(z_meta_train, celltype_meta_train)
            centroids_tr.append(ctr_meta_train)

            meta_train_loss += ce_loss
        
        if meta_train_loss == 0:
            return
        
        FT = self.module.z_encoder
        FT.zero_grad()
        grads_FT = torch.autograd.grad(meta_train_loss, FT.parameters(), create_graph=True, allow_unused=True)
        fast_weights_FT = self.cloned_state_dict(FT)
        adapted_params = OrderedDict()
        for (key, val), grad in zip(FT.named_parameters(), grads_FT):
            adapted_params[key] = val - self.meta_lr * grad
            fast_weights_FT[key] = adapted_params[key]  
        
        CLS = self.celltype_classifier
        CLS.zero_grad()
        grads_CLS = torch.autograd.grad(meta_train_loss, CLS.parameters(), create_graph=True, allow_unused=True)
        fast_weights_CLS = self.cloned_state_dict(CLS)

        adapted_params = OrderedDict()
        for (key, val), grad in zip(CLS.named_parameters(), grads_CLS):
            adapted_params[key] = val - self.meta_lr * grad
            fast_weights_CLS[key] = adapted_params[key]  

        dist_mat_tr = self.row_pairwise_distances(centroids_tr) 

        for j in meta_test_idx:  
            mask = (batch[REGISTRY_KEYS.BATCH_KEY] == j)
            if torch.all(~mask):
                continue
            z_meta_test = z[mask.squeeze(1)]
            cls_logits_meta_test = cls_logits[mask.squeeze(1)]
            celltype_meta_test = batch[REGISTRY_KEYS.LABELS_KEY][mask]

            ctr_meta_test = self.class_centroid(z_meta_test, celltype_meta_test)
            Loss_test += self.criterion_ce(cls_logits_meta_test, celltype_meta_test)
            centroids_val.append(ctr_meta_test)
        
        if Loss_test == 0:
            return
        
        dist_mat_val = self.row_pairwise_distances(centroids_val)
        global_loss = self.global_alignment(dist_mat_tr, dist_mat_val)
        meta_test_loss = Loss_test + self.global_coef * global_loss

        meta_loss = meta_train_loss + meta_test_loss
        meta_loss = self.celltype_constraint_weight * meta_loss
        loss += meta_loss

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
        self.log("Domain_meta_learning_loss", meta_loss, on_epoch=True, prog_bar=True)
        self.compute_and_log_metrics(scvi_loss, self.train_metrics, "train")
        opt1.zero_grad()
        opt2.zero_grad()
        self.manual_backward(loss,retain_graph=True)
        opt1.step()
        opt2.step()

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
        params2 = filter(lambda p: p.requires_grad, self.celltype_classifier.parameters())
        optimizer2 = torch.optim.Adam(
            params2, lr=1e-3, eps=0.01, weight_decay=self.weight_decay
        )
        config2 = {"optimizer": optimizer2}
        opts = [config1.pop("optimizer"), config2["optimizer"]]

        if "lr_scheduler" in config1:
                scheds = [config1["lr_scheduler"]]
                return opts, scheds
        else:
                return opts