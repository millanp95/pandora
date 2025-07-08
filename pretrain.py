#!/usr/bin/env python

import builtins
import math
import os
import shutil
import time
from datetime import datetime
from socket import gethostname

import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim
from torch import nn
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from src.datasets import MaskedOneHotDataset, data_from_df, HDF5MaskedDataset
from src.models import CNN_MLM
from src.utils import (
    load_pretrained_model,
    safe_save_model,
    setup_slurm_distributed,
    check_is_distributed,
    get_num_cpu_available,
)

BASE_BATCH_SIZE = 128


import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader


def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl")


def run(config):
   
    # DDP setup =====================================
    setup_slurm_distributed()
    config.world_size   = int(os.environ.get("WORLD_SIZE", 1))
    config.distributed  = check_is_distributed()
    
    if config.world_size > 1 and not config.distributed:
        raise EnvironmentError(
            f"WORLD_SIZE is {config.world_size}, but not all other required"
            " environment variables for distributed training are set."
        )    
    # Work out the total batch size depending on the number of GPUs we are using
    config.batch_size = config.batch_size_per_gpu * config.world_size

    # Device =====================================
    if config.distributed:
        # For multiprocessing distributed training, gpu rank needs to be
        # set to the global rank among all the processes.
        config.global_rank = int(os.environ["RANK"])
        config.local_rank = int(os.environ["LOCAL_RANK"])
        print(
            f"Rank {config.global_rank} of {config.world_size} on {gethostname()}"
            f" (local GPU {config.local_rank} of {torch.cuda.device_count()})."
            f" Communicating with master at {os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
        )
        dist.init_process_group(backend="nccl")
    else:
        config.global_rank = 0

    # Suppress printing if this is not the master process for the node
    if config.distributed and config.global_rank != 0:

        def print_pass(*args, **kwargs):
            pass

        builtins.print = print_pass

    print()
    print("Configuration:")
    print()
    print(config)
    print()
    print(f"Found {torch.cuda.device_count()} GPUs and {get_num_cpu_available()} CPUs.")

    # Check which device to use
    use_cuda = not config.no_cuda and torch.cuda.is_available()

    if config.distributed and not use_cuda:
        raise EnvironmentError("Distributed training with NCCL requires CUDA.")
    if not use_cuda:
        raise EnvironmentError("Pretraining requires CUDA.")
    elif config.local_rank is not None:
        device = f"cuda:{config.local_rank}"
    else:
        device = "cuda"

    print(f"Using device {device}", flush=True)

    # W&B init (master only) =====================================
    run_name = (f"{config.run_name}__{config.run_id}" if config.run_name and config.run_id
            else None
        )
    
    if config.log_wandb and config.global_rank == 0:
        # Lazy import of wandb, since logging to wandb is optional
        import wandb
   
        wandb.init(
            name=run_name,
            id=config.run_id,
            resume="allow",
            group=config.run_id,
            entity=config.wandb_entity,
            project=config.wandb_project,
            config=vars(config),
            job_type="pretrain",
        )
        wandb.config.update({"checkpoint_path": config.checkpoint_path}, allow_val_change=True)

    # DataLoader with DistributedSampler =====================================
    h5_file = os.path.join(config.data_dir, "pre_training.h5")
    dataset = HDF5MaskedDataset(
        h5_file,
        mask_ratio=config.mask_ratio,
        token_size=config.chunk_size,
        randomize_offset=not config.no_offset,
    )
    sampler = DistributedSampler(dataset, shuffle=True) if config.distributed else None
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.cpu_workers or get_num_cpu_available(),
        pin_memory=True,
        prefetch_factor=2,
        drop_last=True,
    )

    # Model, optimizer, scheduler, loss =====================================
    # if there is no checkpoint provided, the checkpoint is fully determined by 
    # the run name 

    if not config.checkpoint_path:
        config.checkpoint_path = f"model_checkpoints/{run_name}_checkpoint.pt"
    
    
    if os.path.exists(config.checkpoint_path):
        print(f"Loading training snapshot from {config.checkpoint_path}")
        model, ckpt = load_pretrained_model(config.checkpoint_path)
        current_epoch = ckpt["epoch"]
    else:
        model = CNN_MLM(max_len=config.max_len)
        current_epoch = 1
    
    
    model = CNN_MLM(max_len=config.max_len).to(device)
    if config.distributed:
        model = DDP(model, device_ids=[config.local_rank])
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr_relative,
        weight_decay=config.weight_decay,
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.lr_relative,
        epochs=config.epochs,
        steps_per_epoch=len(loader),
        pct_start=0.3,
        anneal_strategy="linear",
    )
    criterion = nn.CrossEntropyLoss()

    # Training loop =====================================
    for epoch in range(current_epoch, config.epochs + 1):
        if sampler:
            sampler.set_epoch(epoch)

        # accumulators
        sum_lm = sum_ls = sum_am = sum_as = 0.0

        for step, (x_masked, targets, att_mask, mask) in enumerate(loader, start=1):
            x_masked = x_masked.to(device)
            targets  = targets.to(device)
            mask      = mask.to(device)
            att_mask  = att_mask.to(device)

            logits = model(x_masked, att_mask)      # (B, L, 4)
            preds  = logits.argmax(dim=-1)          # (B, L)

            valid_pos  = (targets >= 0) & (targets < 4)
            masked_pos = mask & valid_pos
            seen_pos   = (~mask) & valid_pos

            lm = (
                criterion(logits[masked_pos], targets[masked_pos])
                if masked_pos.any()
                else torch.tensor(0.0, device=device)
            )
            ls = (
                criterion(logits[seen_pos], targets[seen_pos])
                if seen_pos.any()
                else torch.tensor(0.0, device=device)
            )
            loss = config.weight_mask * lm + (1 - config.weight_mask) * ls

            # average across GPUs
            if config.distributed:
                dist.reduce(lm,  0, op=dist.ReduceOp.AVG)
                dist.reduce(ls,  0, op=dist.ReduceOp.AVG)
                dist.reduce(loss,0, op=dist.ReduceOp.AVG)

            lm_val = lm.item()
            ls_val = ls.item()

            acc_m = (
                (preds[masked_pos] == targets[masked_pos]).float().mean()
                if masked_pos.any()
                else torch.tensor(0.0, device=device)
            )
            acc_s = (
                (preds[seen_pos] == targets[seen_pos]).float().mean()
                if seen_pos.any()
                else torch.tensor(0.0, device=device)
            )
            acc_o = (
                (preds[valid_pos] == targets[valid_pos]).float().mean()
                if valid_pos.any()
                else torch.tensor(0.0, device=device)
            )

            if config.distributed:
                dist.reduce(acc_m, 0, op=dist.ReduceOp.AVG)
                dist.reduce(acc_s, 0, op=dist.ReduceOp.AVG)
                dist.reduce(acc_o, 0, op=dist.ReduceOp.AVG)

            acc_m_val = 100 * acc_m.item()
            acc_s_val = 100 * acc_s.item()
            acc_o_val = 100 * acc_o.item()

            # debug print first batch
            if epoch == 1 and step == 1 and config.global_rank == 0:
                print("=== DEBUG BATCH ===")
                print("x_masked.shape:", x_masked.shape)
                print("att_mask.shape:", att_mask.shape)
                print("targets.shape:", targets.shape)
                print("mask.shape:   ", mask.shape)
                print("logits.shape: ", logits.shape)
                print("preds[0]:     ", preds[0])
                print("targets[0]:   ", targets[0])
                print("mask[0]:      ", mask[0])
                print("===================")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # accumulate epoch stats
            sum_lm += lm_val
            sum_ls += ls_val
            sum_am += acc_m_val
            sum_as += acc_s_val

            # print & wandb.log
            if config.global_rank == 0 and step % config.print_interval == 0:
                print(
                    f"Epoch {epoch} Step {step}/{len(loader)} | "
                    f"Lm={lm_val:.4f} Ls={ls_val:.4f} Tot={loss.item():.4f} | "
                    f"Am={acc_m_val:.2f}% As={acc_s_val:.2f}%"
                )
            if config.log_wandb and config.global_rank == 0 and step % config.log_interval == 0:
                wandb.log({
                    "train/loss_masked": lm_val,
                    "train/loss_seen":   ls_val,
                    "train/loss":        loss.item(),
                    "train/acc_masked":  acc_m_val,
                    "train/acc_seen":    acc_s_val,
                    "train/acc_overall": acc_o_val,
                    "train/epoch":       epoch,
                    "train/step":        step,
                    "train/epoch_progress": epoch - 1 + step / len(loader),
                })

        # end of epoch summary & save (master only)
        if config.global_rank == 0:
            avg_lm = sum_lm / len(loader)
            avg_ls = sum_ls / len(loader)
            avg_am = sum_am / len(loader)
            avg_as = sum_as / len(loader)
            print(
                f">>> Epoch {epoch} DONE | "
                f"MeanLm={avg_lm:.4f} MeanLs={avg_ls:.4f} | "
                f"MeanAm={avg_am:.2f}% MeanAs={avg_as:.2f}%"
            )
            safe_save_model(
                {"model": model, "optimizer": optimizer, "scheduler": scheduler},
                config.checkpoint_path,
                config=config,
                epoch=epoch,
            )



def get_parser():
    r"""
    Build argument parser for the command line interface.

    Returns
    -------
    parser : argparse.ArgumentParser
        CLI argument parser.
    """
    import argparse
    import sys

    # Use the name of the file called to determine the name of the program
    prog = os.path.split(sys.argv[0])[1]
    if prog == "__main__.py" or prog == "__main__":
        # If the file is called __main__.py, go up a level to the module name
        prog = os.path.split(__file__)[1]
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Pretrain CNN_MLM.",
        add_help=False,
    )
    # Help arg ----------------------------------------------------------------
    group = parser.add_argument_group("Help")
    group.add_argument(
        "--help",
        "-h",
        action="help",
        help="Show this help message and exit.",
    )
    # Dataset args ------------------------------------------------------------
    group = parser.add_argument_group("Dataset")
    group.add_argument(
        "--dataset",
        dest="dataset_name",
        type=str,
        default="CANADA-1.5M",
        help="Name of the dataset to learn. Default: %(default)s",
    )
    group.add_argument(
        "--data_dir",
        "--data-dir",
        dest="data_dir",
        type=str,
        default=None,
        help="Directory within which the dataset can be found. Default is the directory <repo>/data.",
    )
    group.add_argument(
        "--kernel_size",
        type=int,
        default=4,
        help="Kernel size to use in the input CNN for DNA tokenization. Default: %(default)s",
    )
    group.add_argument(
        "--stride",
        type=int,
        default=2,
        help="Stride to use for in the input CNN for DNA tokenization. Default: %(default)s",
    )
    group.add_argument(
        "--max-len",
        "--max_len",
        type=int,
        default=660,
        help="Maximum length of input sequences. Default: %(default)s",
    )
    # Architecture args -------------------------------------------------------
    group.add_argument(
        "--n-layers",
        "--n_layers",
        type=int,
        default=3,
        help="Number of layers in the transformer. Default: %(default)s",
    )
    group.add_argument(
        "--n-heads",
        "--n_heads",
        type=int,
        default=4,
        help="Number of attention heads in the transformer. Default: %(default)s",
    )
    # MLM args -------------------------------------------------------
    group = parser.add_argument_group("MLM parameters")
    group.add_argument(
        "--mask-ratio",
        "--masking-ratio",
        "--masking_ratio",
        "--mask_ratio",
        dest="mask_ratio",
        type=float,
        default=0.5,
        help="Proportion of tokens to be masked in the MLM. Default: %(default)s",
    )
    group.add_argument(
        "--chunk-size",
        "--chunk_size",
        type=int,
        default=4,
        help="Number of consecutive tokens to be masked. Default: %(default)s",
    )
    group.add_argument(
        "--no_offset",
        action="store_true",
        help="Do not use offset for the augmentations",
    )
    group.add_argument(
        "--weight-mask",
        "--weight_mask",
        type=float,
        default=0.95,
        help="Loss penalty term for the masking tokens",
    )
    # Optimization args -------------------------------------------------------
    group = parser.add_argument_group("Optimization routine")

    group.add_argument(
        "--epochs",
        type=int,
        default=35,
        help="Number of epochs to train for. Default: %(default)s",
    )
    group.add_argument(
        "--lr",
        dest="lr_relative",
        type=float,
        default=0.0001,
        help=(
            f"Maximum learning rate, set per {BASE_BATCH_SIZE} batch size."
            " The actual learning rate used will be scaled up by the total"
            " batch size (across all GPUs). Default: %(default)s"
        ),
    )
    group.add_argument(
        "--weight-decay",
        "--weight_decay",
        "--wd",
        dest="weight_decay",
        type=float,
        default=1e-7,
        help="Weight decay. Default: %(default)s",
    )
    group.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help="Name of optimizer (case-sensitive). Default: %(default)s",
    )
    group.add_argument(
        "--scheduler",
        type=str,
        default="OneCycle",
        help="Learning rate scheduler. Default: %(default)s",
    )
    # Output checkpoint args --------------------------------------------------
    group = parser.add_argument_group("Output checkpoint")
    group.add_argument(
        "--models-dir",
        type=str,
        default="model_checkpoints",
        metavar="PATH",
        help="Output directory for all models. Ignored if --checkpoint is set. Default: %(default)s",
    )
    group.add_argument(
        "--checkpoint",
        dest="checkpoint_path",
        default="",
        type=str,
        metavar="PATH",
        help=(
            "Save and resume partially trained model and optimizer state from this checkpoint."
            " Overrides --models-dir."
        ),
    )
    group.add_argument(
        "--save-best-model",
        action="store_true",
        help="Save a copy of the model with best validation performance.",
    )
    # Reproducibility args ----------------------------------------------------
    group = parser.add_argument_group("Reproducibility")
    group.add_argument(
        "--seed",
        type=int,
        help="Random number generator (RNG) seed. Default: not controlled",
    )
    group.add_argument(
        "--deterministic",
        action="store_true",
        help="Disable non-deterministic features of cuDNN.",
    )
    # Hardware configuration args ---------------------------------------------
    group = parser.add_argument_group("Hardware configuration")
    group.add_argument(
        "--batch-size",
        "--batch_size",
        dest="batch_size_per_gpu",
        type=int,
        default=16,
        help=(
            "Batch size per GPU. The total batch size will be this value times"
            " the total number of GPUs used. Default: %(default)s"
        ),
    )
    group.add_argument(
        "--cpu-workers",
        "--cpu_workers",
        "--workers",
        dest="cpu_workers",
        type=int,
        help="Number of CPU workers per node. Default: number of CPUs available on device.",
    )
    group.add_argument(
        "--no-cuda",
        action="store_true",
        help="Use CPU only, no GPUs.",
    )
    group.add_argument(
        "--gpu",
        "--local-rank",
        dest="local_rank",
        default=None,
        type=int,
        help="Index of GPU to use when training a single process. (Ignored for distributed training.)",
    )
    # Logging args ------------------------------------------------------------
    group = parser.add_argument_group("Debugging and logging")
    group.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Number of batches between each log to wandb (if enabled). Default: %(default)s",
    )
    group.add_argument(
        "--print-interval",
        type=int,
        default=100,
        help="Number of batches between each print to STDOUT. Default: same as LOG_INTERVAL.",
    )
    group.add_argument(
        "--log-wandb",
        action="store_true",
        help="Log results with Weights & Biases https://wandb.ai",
    )
    group.add_argument(
        "--disable-wandb",
        "--disable_wandb",
        "--no-wandb",
        dest="disable_wandb",
        action="store_true",
        help="Overrides --log-wandb and ensures wandb is always disabled.",
    )
    group.add_argument(
        "--wandb-entity",
        type=str,
        default="uoguelph_mlrg",
        help="The entity (organization) within which your wandb project is located. Default: %(default)s",
    )
    group.add_argument(
        "--wandb-project",
        type=str,
        default="CNN_MLM",
        help="Name of project on wandb, where these runs will be saved. Default: %(default)s",
    )
    group.add_argument(
        "--run-name",
        type=str,
        help="Human-readable identifier for the model run or job. Used to name the run on wandb.",
    )
    group.add_argument(
        "--run-id",
        type=str,
        help="Unique identifier for the model run or job. Used as the run ID on wandb.",
    )
    return parser


def cli():
    r"""Command-line interface for model training."""
    parser = get_parser()
    config = parser.parse_args()
    # If stride value is ommited., then it is equal to k_mer
    if not config.stride:
        config.stride = config.k_mer
    # Handle disable_wandb overriding log_wandb and forcing it to be disabled.
    if config.disable_wandb:
        config.log_wandb = False
    del config.disable_wandb

    #ddp_setup()
    run(config)
    destroy_process_group()


if __name__ == "__main__":
    cli()
