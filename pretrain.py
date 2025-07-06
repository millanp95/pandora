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
    """
    Training loop that:
      - takes full-length logits from model(x_masked, att_mask)
      - splits masked vs. seen losses (weighted by weight_mask)
      - logs per-100-step losses
      - uses OneCycleLR
    Assumes:
      MaskedOneHotDataset returns (x_masked, targets, att_mask, mask)
      with x_masked: (B, L, 5),
           targets:  (B, L) in {-1,0,1,2,3,4},
           att_mask: (B, L),
           mask:     (B, L) Boolean for masked A/C/G/T only.
      model(x_masked, att_mask) -> logits (B, L, 4).
    """

    if config.log_wandb:
        # Lazy import of wandb, since logging to wandb is optional
        import wandb

    # DISTRIBUTION ============================================================
    # Setup for distributed training
    setup_slurm_distributed()
    config.world_size = int(os.environ.get("WORLD_SIZE", 1))
    config.distributed = check_is_distributed()

    if config.world_size > 1 and not config.distributed:
        raise EnvironmentError(
            f"WORLD_SIZE is {config.world_size}, but not all other required"
            " environment variables for distributed training are set."
        )
    # Work out the total batch size depending on the number of GPUs we are using
    config.batch_size = config.batch_size_per_gpu * config.world_size

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

    weight_mask = 0.5

    #df = pd.read_csv(os.path.join(config.data_dir, "pre_training.csv"))
    #X, _ = data_from_df(df)

    #dataset = MaskedOneHotDataset(X, mask_ratio=config.mask_ratio, chunk_size=4)
    #loader = DataLoader(
    #    dataset,
    #    batch_size=config.batch_size_per_gpu,
    #   shuffle=False,
    #    num_workers=4,
    #    sampler=DistributedSampler(dataset),
    #)

    file_name = os.path.join(config.data_dir, "unseen.h5")
    print(f"Pretrain File: {file_name}")
    dataset = HDF5MaskedDataset(file_name, mask_ratio=0.5, token_size=4)
    sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size_per_gpu * config.world_size,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
        sampler = sampler,
        drop_last = True
    )

    if os.path.exists(config.checkpoint_path):
        print(f"Loading training snapshot from {config.checkpoint_path}")
        model, ckpt = load_pretrained_model(config.checkpoint_path)
        current_epoch = ckpt["epoch"]
    else:
        model = CNN_MLM(max_len=config.max_len)
        current_epoch = 1

    model.to(device)
    model = DDP(model, device_ids=[device])

    optimizer = optim.AdamW(model.parameters(), lr=config.lr_relative)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.lr_relative,
        epochs=config.epochs,
        steps_per_epoch=len(loader),
        pct_start=0.3,
        anneal_strategy="linear",
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(current_epoch, config.epochs + 1):
        total_loss_masked = total_acc_masked = count_masked = 0
        total_loss_seen = total_acc_seen = count_seen = 0
        loader.sampler.set_epoch(epoch)
        for step, (x_masked, targets, att_mask, mask) in enumerate(loader, start=1):
            # --- Move to GPU ---
            x_masked = x_masked.to(device)  # (B, L, 5)
            targets = targets.to(device)  # (B, L)
            att_mask = att_mask.to(device)  # (B, L)
            mask = mask.to(device)  # (B, L)

            # --- Forward ---
            logits = model(x_masked, att_mask)  # (B, L, 4)
            #print(logits.shape)

            preds = logits.argmax(dim=-1)  # (B, L)

            # --- Build masks for loss ---
            valid_pos = (targets >= 0) & (targets < 4)
            masked_pos = mask & valid_pos
            seen_pos = (~mask) & valid_pos

            # --- Compute losses ---
            loss_masked = (
                criterion(logits[masked_pos], targets[masked_pos])
                if masked_pos.any()
                else torch.tensor(0.0, device=device)
            )
            loss_seen = (
                criterion(logits[seen_pos], targets[seen_pos])
                if seen_pos.any()
                else torch.tensor(0.0, device=device)
            )
            loss = weight_mask * loss_masked + loss_seen
            print(loss)

            # --- Backprop & step ---
            optimizer.zero_grad()
            loss.backward()
            # clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # --- Accumulate metrics ---
            if masked_pos.any():
                total_acc_masked += (
                    (preds[masked_pos] == targets[masked_pos]).sum().item()
                )
                total_loss_masked += loss_masked.item() * masked_pos.sum().item()
                count_masked += masked_pos.sum().item()

            if seen_pos.any():
                total_acc_seen += (preds[seen_pos] == targets[seen_pos]).sum().item()
                total_loss_seen += loss_seen.item() * seen_pos.sum().item()
                count_seen += seen_pos.sum().item()

            # --- Log every 100 steps ---
            if step % 100 == 0:
                print(
                    f"Epoch {epoch} Step {step}/{len(loader)} | "
                    f"Loss_masked: {loss_masked.item():.4f} | "
                    f"Loss_seen: {loss_seen.item():.4f} | "
                    f"Total: {loss.item():.4f}"
                )

        # --- Epoch summary ---
        avg_loss_masked = total_loss_masked / count_masked if count_masked else 0.0
        avg_acc_masked = (
            100.0 * total_acc_masked / count_masked if count_masked else 0.0
        )
        avg_loss_seen = total_loss_seen / count_seen if count_seen else 0.0
        avg_acc_seen = 100.0 * total_acc_seen / count_seen if count_seen else 0.0

        print(
            f"Epoch {epoch}/{config.epochs} DONE ➞ "
            f"Masked Loss: {avg_loss_masked:.4f}, Acc: {avg_acc_masked:.2f}% | "
            f"Seen Loss: {avg_loss_seen:.4f}, Acc: {avg_acc_seen:.2f}%"
        )
        if config.global_rank == 0: 
            safe_save_model(
            {
                "model": model,
                "optimizer": optimizer,
                "scheduler": scheduler,
            },
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
        default=None,
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
