"""Collection of PyTorchLightning callback functions"""

import sys
import lightning as pl
import tabulate
import torch
from lightning.pytorch.callbacks import RichModelSummary as PLRichModelSummary
from lightning.pytorch.callbacks import TQDMProgressBar as PLTQDMProgressBar
from lightning.pytorch.callbacks.progress.tqdm_progress import Tqdm
from . import dist_utils as utils


class TQDMProgressBar(PLTQDMProgressBar):
    """TQDM Callback"""

    original_test_results: dict = None

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Every end of train epoch -> display a summary Table including `self.callback_metrics` and `self.original_test_results`"""
        super().on_train_epoch_end(trainer, pl_module)
        if utils.is_main_process():
            callback_metrics: dict = dict(trainer.callback_metrics)
            if self.original_test_results is not None:
                callback_metrics.update(self.original_test_results)
            _max_key_len: int = max(map(len, callback_metrics.keys()))
            keys: list = [f"{k: <{_max_key_len}}" for k in callback_metrics.keys()]
            values: list = list(callback_metrics.values())
            _table: str = tabulate.tabulate(
                tabular_data=zip(keys, values), headers=["Metric", "Value"], tablefmt="mixed_outline"
            )
            trainer.print(f"\n\nEpoch {trainer.current_epoch} summary:\n{_table}\n")

    def init_validation_tqdm(self) -> Tqdm:
        return Tqdm(
            desc=f"{self.validation_description} [Epoch {self.trainer.current_epoch}]",
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=False,
            file=sys.stdout,
            bar_format=self.BAR_FORMAT,
        )


class RichModelSummary(PLRichModelSummary):
    """Model Summary Table Callback"""

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self._max_depth:
            return
        model_summary = self._summary(trainer, pl_module)
        summary_data = model_summary._get_summary_data()
        num_of_trainable_parameters = sum(
            [
                torch.prod(torch.tensor(p.shape)).item()
                for p in pl_module._model.parameters(recurse=True)
                if p.requires_grad
            ]
        )
        num_of_non_trainable_parameters = sum(
            [torch.prod(torch.tensor(p.shape)).item() for p in pl_module._model.buffers(recurse=True)]
        ) + sum(
            [
                torch.prod(torch.tensor(p.shape)).item()
                for p in pl_module._model.parameters(recurse=True)
                if not p.requires_grad
            ]
        )
        total_parameters = int(num_of_trainable_parameters + num_of_non_trainable_parameters)
        trainable_parameters = int(num_of_trainable_parameters)
        model_size = model_summary.model_size
        if trainer.is_global_zero:
            self.summarize(summary_data, total_parameters, trainable_parameters, model_size, **self._summarize_kwargs)
