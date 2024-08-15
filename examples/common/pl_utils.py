"""Collection of PyTorchLightning helper functions"""

import os
from pathlib import Path
from typing import Literal, Optional

import lightning as pl
import torch.distributed as dist
from lightning.pytorch.tuner import Tuner

from . import dist_utils as utils


def tune_batch_size(
    trainer: pl.Trainer,
    module: pl.LightningModule,
    data_module: pl.LightningDataModule,
    output_dir: str,
    batch_arg_name: str,
    method: Literal["fit", "validate", "test", "predict"],
    max_trials: int,
    steps_per_trial: int,
    init_val: int = 1,
) -> int:
    tuner = Tuner(trainer)
    optimal_batch_size: Optional[int] = tuner.scale_batch_size(
        model=module,
        datamodule=data_module,
        method=method,
        max_trials=max_trials,
        steps_per_trial=steps_per_trial,
        init_val=init_val,
        mode="binsearch",
        batch_arg_name=batch_arg_name,  # it will look for this attribute in either 'module' or 'datamodule'
    )
    if utils.is_dist_avail_and_initialized():
        dist.barrier()
        if optimal_batch_size is not None:
            all_selected: list = [None] * utils.get_world_size()
            dist.all_gather_object(object_list=all_selected, obj=optimal_batch_size)
            optimal_batch_size: int = min(all_selected)
    if optimal_batch_size is not None:
        if method == "test":
            optimal_batch_size = max(1, int(optimal_batch_size * 0.75))
        else:
            optimal_batch_size = max(1, int(optimal_batch_size * 0.8))
        setattr(data_module, batch_arg_name, optimal_batch_size)
    if utils.is_main_process():
        for _p in Path(output_dir).rglob(f".scale_batch_size_*"):
            os.remove(_p)
    return optimal_batch_size
