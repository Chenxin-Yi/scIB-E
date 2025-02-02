from __future__ import annotations

from lightning import LightningDataModule

from scvi.dataloaders import DataSplitter, KFoldDataSplitter
from scvi.model._utils import get_max_epochs_heuristic, use_distributed_sampler
from scvi.train import TrainingPlan, TrainRunner
from scvi.utils._docstrings import devices_dsp
from scvi.train.batch_correction_trainingplans import *


class UnsupervisedTrainingMixin:
    """General purpose unsupervised train method."""

    _data_splitter_cls = DataSplitter
    _training_plan_cls = TrainingPlan
    _train_runner_cls = TrainRunner

    def load_training_plan_cls(self, method_name):
        method_name = method_name+"_TrainingPlan"
        if method_name in globals():
            batch_correction_method = eval(method_name)
            return batch_correction_method
        else:
            raise ValueError(f"The training plan of '{method_name}' is not found.")

    @devices_dsp.dedent
    def train(
        self,
        max_epochs: int | None = None,
        accelerator: str = "auto",
        devices: int | list[int] | str = "auto",
        train_size: float = 0.9,
        validation_size: float | None = None,
        shuffle_set_split: bool = True,
        load_sparse_tensor: bool = False,
        batch_size: int = 128,
        early_stopping: bool = False,
        datasplitter_kwargs: dict | None = None,
        batch_correction_method: str | None = None,
        batch_correction_weight: float = 1.0,
        plan_kwargs: dict | None = None,
        data_module: LightningDataModule | None = None,
        **trainer_kwargs,
    ):
        """Train the model.

        Parameters
        ----------
        max_epochs
            The maximum number of epochs to train the model. The actual number of epochs may be
            less if early stopping is enabled. If ``None``, defaults to a heuristic based on
            :func:`~scvi.model.get_max_epochs_heuristic`. Must be passed in if ``data_module`` is
            passed in, and it does not have an ``n_obs`` attribute.
        %(param_accelerator)s
        %(param_devices)s
        train_size
            Size of training set in the range ``[0.0, 1.0]``. Passed into
            :class:`~scvi.dataloaders.DataSplitter`. Not used if ``data_module`` is passed in.
        validation_size
            Size of the test set. If ``None``, defaults to ``1 - train_size``. If
            ``train_size + validation_size < 1``, the remaining cells belong to a test set. Passed
            into :class:`~scvi.dataloaders.DataSplitter`. Not used if ``data_module`` is passed in.
        shuffle_set_split
            Whether to shuffle indices before splitting. If ``False``, the val, train, and test set
            are split in the sequential order of the data according to ``validation_size`` and
            ``train_size`` percentages. Passed into :class:`~scvi.dataloaders.DataSplitter`. Not
            used if ``data_module`` is passed in.
        load_sparse_tensor
            ``EXPERIMENTAL`` If ``True``, loads data with sparse CSR or CSC layout as a
            :class:`~torch.Tensor` with the same layout. Can lead to speedups in data transfers to
            GPUs, depending on the sparsity of the data. Passed into
            :class:`~scvi.dataloaders.DataSplitter`. Not used if ``data_module`` is passed in.
        batch_size
            Minibatch size to use during training. Passed into
            :class:`~scvi.dataloaders.DataSplitter`. Not used if ``data_module`` is passed in.
        early_stopping
            Perform early stopping. Additional arguments can be passed in through ``**kwargs``.
            See :class:`~scvi.train.Trainer` for further options.
        datasplitter_kwargs
            Additional keyword arguments passed into :class:`~scvi.dataloaders.DataSplitter`.
            Values in this argument can be overwritten by arguments directly passed into this
            method, when appropriate. Not used if ``data_module`` is passed in.
        plan_kwargs
            Additional keyword arguments passed into :class:`~scvi.train.TrainingPlan`. Values in
            this argument can be overwritten by arguments directly passed into this method, when
            appropriate.
        data_module
            ``EXPERIMENTAL`` A :class:`~lightning.pytorch.core.LightningDataModule` instance to use
            for training in place of the default :class:`~scvi.dataloaders.DataSplitter`. Can only
            be passed in if the model was not initialized with :class:`~anndata.AnnData`.
        **kwargs
           Additional keyword arguments passed into :class:`~scvi.train.Trainer`.
        """
        if data_module is not None and not self._module_init_on_train:
            raise ValueError(
                "Cannot pass in `data_module` if the model was initialized with `adata`."
            )
        elif data_module is None and self._module_init_on_train:
            raise ValueError(
                "If the model was not initialized with `adata`, a `data_module` must be passed in."
            )

        if max_epochs is None:
            if data_module is None:
                max_epochs = get_max_epochs_heuristic(self.adata.n_obs)
            elif hasattr(data_module, "n_obs"):
                max_epochs = get_max_epochs_heuristic(data_module.n_obs)
            else:
                raise ValueError(
                    "If `data_module` does not have `n_obs` attribute, `max_epochs` must be "
                    "passed in."
                )

        if data_module is None:
            datasplitter_kwargs = datasplitter_kwargs or {}
            if 'KFold' in trainer_kwargs:
                self._data_splitter_cls = KFoldDataSplitter
                data_module = self._data_splitter_cls(
                adata_manager=self.adata_manager,
                n_splits=trainer_kwargs['KFold'],
                shuffle=True,
                load_sparse_tensor=False,
                pin_memory=False
            )
            else:
                data_module = self._data_splitter_cls(
                    self.adata_manager,
                    train_size=train_size,
                    validation_size=validation_size,
                    batch_size=batch_size,
                    shuffle_set_split=shuffle_set_split,
                    distributed_sampler=use_distributed_sampler(trainer_kwargs.get("strategy", None)),
                    load_sparse_tensor=load_sparse_tensor,
                    **datasplitter_kwargs,
                )
        elif self.module is None:
            self.module = self._module_cls(
                data_module.n_vars,
                n_batch=data_module.n_batch,
                n_labels=getattr(data_module, "n_labels", 1),
                n_continuous_cov=getattr(data_module, "n_continuous_cov", 0),
                n_cats_per_cov=getattr(data_module, "n_cats_per_cov", None),
                **self._module_kwargs,
            )

        plan_kwargs = plan_kwargs or {}

        if batch_correction_method:
            self._training_plan_cls = self.load_training_plan_cls(batch_correction_method)
            training_plan = self._training_plan_cls(self.module, batch_correction_weight=batch_correction_weight, **plan_kwargs)
        else:
            training_plan = self._training_plan_cls(self.module, **plan_kwargs)

        es = "early_stopping"
        trainer_kwargs[es] = (
            early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]
        )
        runner = self._train_runner_cls(
            self,
            training_plan=training_plan,
            data_splitter=data_module,
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            **trainer_kwargs,
        )
        return runner()
