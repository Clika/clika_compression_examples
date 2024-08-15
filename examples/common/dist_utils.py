"""Distributed training functionalities for multiGPU environment"""

import os

import torch
import torch.distributed as dist


def override_pl_strategy(trainer: "lightning.Trainer", args: "argparse.Namespace") -> None:
    from pytorch_lightning.strategies import FSDPStrategy

    if is_dist_avail_and_initialized():
        trainer.strategy.global_rank = get_rank()
        trainer.strategy.local_rank = args.gpu
        trainer.strategy.world_size = get_world_size()
        type(trainer.strategy).is_global_zero = property(fget=lambda self: is_main_process())
        type(trainer.strategy)._determine_device_ids = lambda x: [args.gpu]
        type(trainer.strategy).barrier = FSDPStrategy.barrier  # could be DDPStrategy as well, doesn't matter.
        type(trainer.strategy).broadcast = FSDPStrategy.broadcast


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


class distributed_local_main_first(object):
    def __enter__(self):
        if is_dist_avail_and_initialized():
            if get_rank() != 0:
                dist.barrier()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if is_dist_avail_and_initialized():
            if get_rank() == 0:
                dist.barrier()
            dist.barrier()
