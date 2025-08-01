#!/usr/bin/env python

import builtins
import math
import os
import sys
import shutil
import time
from datetime import datetime
from socket import gethostname

import torch
import torch.distributed as dist
import torch.optim
from torch import nn
from torch.utils.data.distributed import DistributedSampler
from transformers import BertConfig, BertForTokenClassification

sys.path.append(".")


from barcodebert import utils
from barcodebert.datasets import DNADataset
from barcodebert.io import get_project_root, safe_save_model
from barcodebert.knn_probing import run_knn

# Import the fast attention model conditionally to avoid import errors
try:
    from barcodebert.models import FastBertForTokenClassification
    FAST_ATTENTION_AVAILABLE = True
except ImportError:
    FAST_ATTENTION_AVAILABLE = False
    FastBertForTokenClassification = None



BASE_BATCH_SIZE = 64


def run(config):
    r"""
    Run training job (one worker if using distributed training).

    Parameters
    ----------
    config : argparse.Namespace or OmegaConf
        The configuration for this experiment.
    """
    if config.log_wandb:
        # Lazy import of wandb, since logging to wandb is optional
        import wandb

    if config.seed is not None:
        utils.set_rng_seeds_fixed(config.seed)

    if config.deterministic:
        print("Running in deterministic cuDNN mode. Performance may be slower, but more reproducible.")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # DISTRIBUTION ============================================================
    # Setup for distributed training
    utils.setup_slurm_distributed()
    config.world_size = int(os.environ.get("WORLD_SIZE", 1))
    config.distributed = utils.check_is_distributed()

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
    print(f"Found {torch.cuda.device_count()} GPUs and {utils.get_num_cpu_available()} CPUs.")

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

    # LOAD PRE-EMPTION CHECKPOINT =============================================
    checkpoint = None
    config.model_output_dir = None
    if config.checkpoint_path:
        config.model_output_dir = os.path.dirname(config.checkpoint_path)
    if not config.checkpoint_path:
        # Not trying to resume from a checkpoint
        pass
    elif not os.path.isfile(config.checkpoint_path):
        # Looks like we're trying to resume from the checkpoint that this job
        # will itself create. Let's assume this is to let the job resume upon
        # preemption, and it just hasn't been preempted yet.
        print(f"Skipping premature resumption from preemption: no checkpoint file found at '{config.checkpoint_path}'")
    else:
        print(f"Loading resumption checkpoint '{config.checkpoint_path}'", flush=True)
        # Map model parameters to be load to the specified gpu.
        checkpoint = torch.load(config.checkpoint_path, map_location=device)

    if checkpoint is None:
        # Our epochs go from 1 to n_epoch, inclusive
        start_epoch = 1
    else:
        # Continue from where we left off
        start_epoch = checkpoint["epoch"] + 1
        if config.seed is not None:
            # Make sure we don't get the same behaviour as we did on the
            # first epoch repeated on this resumed epoch.
            utils.set_rng_seeds_fixed(config.seed + start_epoch, all_gpu=False)

    # DATASET =================================================================

    if config.dataset_name not in ["CANADA-1.5M", "BIOSCAN-5M"]:
        raise NotImplementedError(f"Dataset {config.dataset_name} not supported.")

    if config.data_dir is None:
        config.data_dir = os.path.join(get_project_root(), "data")

    dataset_args = {
        "k_mer": config.k_mer,
        "stride": config.stride,
        "max_len": config.max_len,
        "tokenizer": config.tokenizer,
        "tokenize_n_nucleotide": config.tokenize_n_nucleotide,
        "dataset_format": config.dataset_name,
    }

    dataset_train = DNADataset(
        file_path=os.path.join(config.data_dir, "pre_training.csv"),
        randomize_offset=True,
        **dataset_args,
    )
    dataset_val = DNADataset(
        file_path=os.path.join(config.data_dir, "supervised_train.csv"),
        randomize_offset=False,
        **dataset_args,
    )
    eval_set = "Val"

    # Dataloader --------------------------------------------------------------
    dl_train_kwargs = {
        "batch_size": config.batch_size_per_gpu,
        "drop_last": True,
        "sampler": None,
        "shuffle": True,
        "worker_init_fn": utils.worker_seed_fn,
    }
    dl_val_kwargs = {
        "batch_size": config.batch_size_per_gpu,
        "drop_last": False,
        "sampler": None,
        "shuffle": False,
        "worker_init_fn": utils.worker_seed_fn,
    }
    if config.cpu_workers is None:
        config.cpu_workers = utils.get_num_cpu_available()
    if use_cuda:
        cuda_kwargs = {"num_workers": config.cpu_workers, "pin_memory": True}
        dl_train_kwargs.update(cuda_kwargs)
        dl_val_kwargs.update(cuda_kwargs)

    if config.distributed:
        # The DistributedSampler breaks up the dataset across the GPUs
        dl_train_kwargs["sampler"] = DistributedSampler(
            dataset_train,
            shuffle=True,
            seed=config.seed if config.seed is not None else 0,
            drop_last=False,
        )
        dl_train_kwargs["shuffle"] = None
        dl_val_kwargs["sampler"] = DistributedSampler(
            dataset_val,
            shuffle=False,
            drop_last=False,
        )
        dl_val_kwargs["shuffle"] = None

    dataloader_train = torch.utils.data.DataLoader(dataset_train, **dl_train_kwargs)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, **dl_val_kwargs)

    # MODEL ===================================================================
    bert_config = BertConfig(
        vocab_size=dataset_train.vocab_size,
        num_hidden_layers=config.n_layers,
        num_attention_heads=config.n_heads,
        num_labels=4**config.k_mer,
        output_hidden_states=True,
        max_position_embeddings=max(512, math.ceil(1536 / config.stride)),
    )

    # Initializing a model (with random weights) from the bert-base-uncased style configuration
    if getattr(config, 'fast_attention', False):
        if FAST_ATTENTION_AVAILABLE:
            print("Using FastBertForTokenClassification with optimized attention")
            model = FastBertForTokenClassification(bert_config)
        else:
            print("Warning: FastBertForTokenClassification not available, falling back to standard BertForTokenClassification")
            model = BertForTokenClassification(bert_config)
    else:
        model = BertForTokenClassification(bert_config)

    # Configure model for distributed training --------------------------------
    print("\nModel architecture:")
    print(model, flush=True)
    print()

    if not use_cuda:
        print("Using CPU (this will be slow)", flush=True)
    elif config.distributed:
        # Convert batchnorm into SyncBN, using stats computed from all GPUs
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, the DistributedDataParallel
        # constructor should always set a single device scope, otherwise
        # DistributedDataParallel will use all available devices.
        model = model.to(device)
        torch.cuda.set_device(device)
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[config.local_rank], output_device=config.local_rank
        )
    else:
        if config.local_rank is not None:
            torch.cuda.set_device(config.local_rank)
        model = model.to(device)

    # OPTIMIZATION ============================================================
    # Optimizer ---------------------------------------------------------------
    # Set up the optimizer

    # Bigger batch sizes mean better estimates of the gradient, so we can use a
    # bigger learning rate. See https://arxiv.org/abs/1706.02677
    # Hence we scale the learning rate linearly with the total batch size.
    config.lr = config.lr_relative * config.batch_size / BASE_BATCH_SIZE

    # Fetch the constructor of the appropriate optimizer from torch.optim
    optimizer = getattr(torch.optim, config.optimizer)(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    # Scheduler ---------------------------------------------------------------
    # Set up the learning rate scheduler
    if config.scheduler.lower() == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            [p["lr"] for p in optimizer.param_groups],
            epochs=config.epochs,
            steps_per_epoch=len(dataloader_train),
        )
    else:
        raise NotImplementedError(f"Scheduler {config.scheduler} not supported.")

    # Loss function -----------------------------------------------------------
    # Set up loss function
    criterion = nn.CrossEntropyLoss()

    # Mixed Precision ---------------------------------------------------------
    # Set up automatic mixed precision training
    scaler = None
    if config.mixed_precision and use_cuda:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
        print("Mixed precision training enabled with GradScaler.")
    elif config.mixed_precision:
        print("Warning: Mixed precision requested but CUDA is not available. Running in full precision.")

    distance_table = None

    # Mask schedule -----------------------------------------------------------
    if False:
        # Linearly increase to 0.75% masking ratio (disabled)
        # mask_ratios = [(0.75 - 0.25) / config.epochs * x + 0.1 for x in range(config.epochs + 1)]
        raise ValueError("Mask schedule not implemented")
    else:
        # Constant masking rate
        mask_ratios = [0.50 for x in range(config.epochs + 1)]

    # LOGGING =================================================================
    # Setup logging and saving

    # If we're using wandb, initialize the run, or resume it if the job was preempted.
    if config.log_wandb and config.global_rank == 0:
        wandb_run_name = config.run_name
        if wandb_run_name is not None and config.run_id is not None:
            wandb_run_name = f"{wandb_run_name}__{config.run_id}"
        EXCLUDED_WANDB_CONFIG_KEYS = [
            "log_wandb",
            "wandb_entity",
            "wandb_project",
            "global_rank",
            "local_rank",
            "run_name",
            "run_id",
            "model_output_dir",
        ]
        wandb.init(
            name=wandb_run_name,
            id=config.run_id,
            resume="allow",
            group=config.run_id,
            entity=config.wandb_entity,
            project=config.wandb_project,
            config=wandb.helper.parse_config(config, exclude=EXCLUDED_WANDB_CONFIG_KEYS),
            job_type="pretrain",
            tags=["pretrain"],
        )
        # Extra tags for Pandora
        wandb.config.update({
        "xformers": False,
        "cnn":      False,
        "jumbo_cls":False,
        "pos_enc":  "Learned",
        "apex": config.mixed_precision,
        "fast_attention": getattr(config, 'fast_attention', False),
        })
        # If a run_id was not supplied at the command prompt, wandb will
        # generate a name. Let's use that as the run_name.
        if config.run_name is None:
            config.run_name = wandb.run.name
        if config.run_id is None:
            config.run_id = wandb.run.id

    # If we still don't have a run name, generate one from the current time.
    if config.run_name is None:
        config.run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if config.run_id is None:
        config.run_id = utils.generate_id()

    # If no checkpoint path was supplied, automatically determine the path to
    # which we will save the model checkpoint.
    if not config.checkpoint_path and config.models_dir:
        config.model_output_dir = os.path.join(
            config.models_dir,
            config.dataset_name,
            f"{config.run_name}__{config.run_id}",
        )
        config.checkpoint_path = os.path.join(config.model_output_dir, "checkpoint_pretraining.pt")
        if config.log_wandb and config.global_rank == 0:
            wandb.config.update({"checkpoint_path": config.checkpoint_path}, allow_val_change=True)

    # For consistency with finetune, linearprobe and kNN jobs, record the
    # pretrained_run_name and pretrained_run_id.
    if config.log_wandb and config.global_rank == 0:
        wandb.config.update({"pretrained_run_name": config.run_name, "pretrained_run_id": config.run_id})

    if config.checkpoint_path is None:
        print("Model will not be saved.")
    else:
        os.makedirs(config.model_output_dir, exist_ok=True)
        print(f"Model will be saved to '{config.checkpoint_path}'")

    # RESUME ==================================================================
    # Now that everything is set up, we can load the state of the model,
    # optimizer, and scheduler from a checkpoint, if supplied.

    # Initialize step related variables as if we're starting from scratch.
    # Their values will be overridden by the checkpoint if we're resuming.
    total_step = 0
    n_samples_seen = 0

    best_stats = {"max_accuracy": 0, "best_epoch": 0}

    if checkpoint is not None:
        print(f"Loading state from checkpoint (epoch {checkpoint['epoch']})")
        
        # Check model type compatibility
        checkpoint_model_type = checkpoint.get("model_type", "BertForTokenClassification")
        current_model_type = "FastBertForTokenClassification" if getattr(config, 'fast_attention', False) else "BertForTokenClassification"
        
        if checkpoint_model_type != current_model_type:
            print(f"Warning: Model type mismatch! Checkpoint was saved with {checkpoint_model_type}")
            print(f"         but current configuration uses {current_model_type}")
            print("         This may cause compatibility issues.")
        
        total_step = checkpoint["total_step"]
        n_samples_seen = checkpoint["n_samples_seen"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        # Load scaler state if available and we're using mixed precision
        if scaler is not None and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
            print("Loaded scaler state from checkpoint.")
        elif scaler is not None:
            print("Scaler state not found in checkpoint, using fresh scaler.")
        best_stats["max_accuracy"] = checkpoint.get("max_accuracy", 0)
        best_stats["best_epoch"] = checkpoint.get("best_epoch", 0)

    # TRAIN ===================================================================
    print()
    print("Configuration:")
    print()
    print(config, flush=True)
    print()
    
    # Save the number of trainable parameters for logging
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if config.log_wandb and config.global_rank == 0:
        wandb.log({"params": trainable_params})
    
    # Ensure modules are on the correct device
    model = model.to(device)

    # KNN Configuration =======================================================
    # Create configuration for kNN evaluation similar to pandora/pretrain.py
    import argparse
    knn_config = argparse.Namespace()
    knn_config.pretrained_checkpoint_path = config.checkpoint_path
    knn_config.data_dir = config.data_dir
    knn_config.dataset_name = config.dataset_name
    knn_config.taxon = getattr(config, 'knn_taxon', 'genus')  # Use configurable taxonomic level
    knn_config.n_neighbors = 1
    knn_config.metric = "cosine"
        
    # Logging configuration
    knn_config.log_wandb = config.log_wandb and config.global_rank == 0
    knn_config.wandb_entity = getattr(config, 'wandb_entity', None)
    knn_config.wandb_project = getattr(config, 'wandb_project', None)
    knn_config.run_name = getattr(config, 'run_name', None)
    knn_config.run_id = getattr(config, 'run_id', None)
    knn_config.seed = config.seed
    knn_config.deterministic = getattr(config, 'deterministic', False)
    knn_config.disable_wandb = False

    timing_stats = {}
    t_end_epoch = time.time()
    for epoch in range(start_epoch, config.epochs + 1):
        t_start_epoch = time.time()
        if config.seed is not None:
            # If the job is resumed from preemption, our RNG state is currently set the
            # same as it was at the start of the first epoch, not where it was when we
            # stopped training. This is not good as it means jobs which are resumed
            # don't do the same thing as they would be if they'd run uninterrupted
            # (making preempted jobs non-reproducible).
            # To address this, we reset the seed at the start of every epoch. Since jobs
            # can only save at the end of and resume at the start of an epoch, this
            # makes the training process reproducible. But we shouldn't use the same
            # RNG state for each epoch - instead we use the original seed to define the
            # series of seeds that we will use at the start of each epoch.
            epoch_seed = utils.determine_epoch_seed(config.seed, epoch=epoch)
            # We want each GPU to have a different seed to the others to avoid
            # correlated randomness between the workers on the same batch.
            # We offset the seed for this epoch by the GPU rank, so every GPU will get a
            # unique seed for the epoch. This means the job is only precisely
            # reproducible if it is rerun with the same number of GPUs (and the same
            # number of CPU workers for the dataloader).
            utils.set_rng_seeds_fixed(epoch_seed + config.global_rank, all_gpu=False)
            if isinstance(getattr(dataloader_train, "generator", None), torch.Generator):
                # Finesse the dataloader's RNG state, if it is not using the global state.
                dataloader_train.generator.manual_seed(epoch_seed + config.global_rank)
            if isinstance(getattr(dataloader_train.sampler, "generator", None), torch.Generator):
                # Finesse the sampler's RNG state, if it is not using the global RNG state.
                dataloader_train.sampler.generator.manual_seed(config.seed + epoch + 10000 * config.global_rank)

        if hasattr(dataloader_train.sampler, "set_epoch"):
            # Handling for DistributedSampler.
            # Set the epoch for the sampler so that it can shuffle the data
            # differently for each epoch, but synchronized across all GPUs.
            dataloader_train.sampler.set_epoch(epoch)

        # Train ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Note the number of samples seen before this epoch started, so we can
        # calculate the number of samples seen in this epoch.
        n_samples_seen_before = n_samples_seen
        # Run one epoch of training
        train_stats, total_step, n_samples_seen = train_one_epoch(
            config=config,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            dataloader=dataloader_train,
            mask_ratio=mask_ratios[epoch - 1],
            device=device,
            epoch=epoch,
            n_epoch=config.epochs,
            total_step=total_step,
            n_samples_seen=n_samples_seen,
            distance_table=distance_table,
            n_special_tokens=len(dataset_train.special_tokens),
            scaler=scaler,
        )
        t_end_train = time.time()

        timing_stats["train"] = t_end_train - t_start_epoch
        n_epoch_samples = n_samples_seen - n_samples_seen_before
        train_stats["throughput"] = n_epoch_samples / timing_stats["train"]

        print(f"Pretraining epoch {epoch}/{config.epochs} summary:")
        print(f"  Steps ..............{len(dataloader_train):8d}")
        print(f"  Samples ............{n_epoch_samples:8d}")
        if timing_stats["train"] > 172800:
            print(f"  Duration ...........{timing_stats['train']/86400:11.2f} days")
        elif timing_stats["train"] > 5400:
            print(f"  Duration ...........{timing_stats['train']/3600:11.2f} hours")
        elif timing_stats["train"] > 120:
            print(f"  Duration ...........{timing_stats['train']/60:11.2f} minutes")
        else:
            print(f"  Duration ...........{timing_stats['train']:11.2f} seconds")
        print(f"  Throughput .........{train_stats['throughput']:11.2f} samples/sec")
        print(f"  Loss ...............{train_stats['loss']:14.5f}")
        print(f"  Accuracy ...........{train_stats['accuracy']:11.2f} %")
        print(flush=True)

        # Validate ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Evaluate on validation set
        t_start_val = time.time()

        eval_stats = evaluate(
            config=config,
            model=model,
            criterion=criterion,
            dataloader=dataloader_val,
            mask_ratio=mask_ratios[epoch - 1],
            device=device,
            distance_table=distance_table,
            n_special_tokens=len(dataset_val.special_tokens),
        )
        t_end_val = time.time()
        timing_stats["val"] = t_end_val - t_start_val
        eval_stats["throughput"] = len(dataloader_val.dataset) / timing_stats["val"]

        # Check if this is the new best model
        if eval_stats["accuracy"] >= best_stats["max_accuracy"]:
            best_stats["max_accuracy"] = eval_stats["accuracy"]
            best_stats["best_epoch"] = epoch

        print(f"Evaluating epoch {epoch}/{config.epochs} summary:")
        if timing_stats["val"] > 172800:
            print(f"  Duration ...........{timing_stats['val']/86400:11.2f} days")
        elif timing_stats["val"] > 5400:
            print(f"  Duration ...........{timing_stats['val']/3600:11.2f} hours")
        elif timing_stats["val"] > 120:
            print(f"  Duration ...........{timing_stats['val']/60:11.2f} minutes")
        else:
            print(f"  Duration ...........{timing_stats['val']:11.2f} seconds")
        print(f"  Throughput .........{eval_stats['throughput']:11.2f} samples/sec")
        print(f"  Loss ...............{eval_stats['loss']:14.5f}")
        print(f"  Accuracy ...........{eval_stats['accuracy']:11.2f} %")

        # Save model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        t_start_save = time.time()
        if config.model_output_dir and (not config.distributed or config.global_rank == 0):
            save_dict = {
                "model": model,
                "optimizer": optimizer,
                "scheduler": scheduler,
            }
            # Add scaler state if using mixed precision
            if scaler is not None:
                save_dict["scaler"] = scaler
                
            safe_save_model(
                save_dict,
                config.checkpoint_path,
                config=config,
                epoch=epoch,
                total_step=total_step,
                n_samples_seen=n_samples_seen,
                bert_config=bert_config.to_dict(),
                model_type="FastBertForTokenClassification" if getattr(config, 'fast_attention', False) else "BertForTokenClassification",
                **best_stats,
            )
            if config.save_best_model and best_stats["best_epoch"] == epoch:
                ckpt_path_best = os.path.join(config.model_output_dir, "best_pretraining.pt")
                print(f"Copying model to {ckpt_path_best}")
                shutil.copyfile(config.checkpoint_path, ckpt_path_best)

        t_end_save = time.time()
        timing_stats["saving"] = t_end_save - t_start_save

        # kNN Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Run kNN evaluation periodically (similar to pandora/pretrain.py)
        knn_freq = getattr(config, 'knn_eval_frequency', 5)
        if config.global_rank == 0 and knn_freq > 0 and epoch % knn_freq == 1:
            print("Running kNN evaluation...")
            try:
                # Update the checkpoint path for the current epoch
                knn_config.pretrained_checkpoint_path = config.checkpoint_path
                run_knn(knn_config)
            except Exception as e:
                print(f"kNN evaluation failed: {e}")

        # Log to wandb ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Overall time won't include uploading to wandb, but there's nothing
        # we can do about that.
        timing_stats["overall"] = time.time() - t_end_epoch
        t_end_epoch = time.time()

        # Send training and eval stats for this epoch to wandb
        if config.log_wandb and config.global_rank == 0:
            wandb.log(
                {
                    "Pretraining/stepwise/epoch": epoch,
                    "Pretraining/stepwise/epoch_progress": epoch,
                    "Pretraining/stepwise/n_samples_seen": n_samples_seen,
                    "Pretraining/epochwise/epoch": epoch,
                    **{f"Pretraining/epochwise/Train/{k}": v for k, v in train_stats.items()},
                    **{f"Pretraining/epochwise/{eval_set}/{k}": v for k, v in eval_stats.items()},
                    **{f"Pretraining/epochwise/duration/{k}": v for k, v in timing_stats.items()},
                },
                step=total_step,
            )
            # Record the wandb time as contributing to the next epoch
            timing_stats = {"wandb": time.time() - t_end_epoch}
        else:
            # Reset timing stats
            timing_stats = {}
        # Print with flush=True forces the output buffer to be printed immediately
        print(flush=True)

    if start_epoch > config.epochs:
        print("Pretraining already completed!")
    else:
        print(f"Pretraining complete! (Trained epochs {start_epoch} to {config.epochs})")
    print(
        f"Best {eval_set} accuracy was {best_stats['max_accuracy']:.2f}%,"
        f" seen at the end of epoch {best_stats['best_epoch']}",
        flush=True,
    )


def train_one_epoch(
    config,
    model,
    optimizer,
    scheduler,
    criterion,
    dataloader,
    mask_ratio=0.5,
    device="cuda",
    epoch=1,
    n_epoch=None,
    total_step=0,
    n_samples_seen=0,
    distance_table=None,
    n_special_tokens=2,
    scaler=None,
):
    r"""
    Train the encoder and classifier for one epoch.

    Parameters
    ----------
    config : argparse.Namespace or OmegaConf
        The global config object.
    model : torch.nn.Module
        The encoder/decoder network.
    optimizer : torch.optim.Optimizer
        The optimizer.
    scheduler : torch.optim.lr_scheduler._LRScheduler
        The learning rate scheduler.
    criterion : torch.nn.Module
        The loss function.
    dataloader : torch.utils.data.DataLoader
        A dataloader for the training set.
    mask_ratio : float, default=0.5
        The ratio of tokens to mask out.
    device : str or torch.device, default="cuda"
        The device to use.
    epoch : int, default=1
        The current epoch number (indexed from 1).
    n_epoch : int, optional
        The total number of epochs scheduled to train for.
    total_step : int, default=0
        The total number of steps taken so far.
    n_samples_seen : int, default=0
        The total number of samples seen so far.
    distance_table : torch.Tensor, optional
        A pre-computed table of edit distances (irrelevant).
    n_special_tokens: int, default=2
        Number of special (non-kmer) tokens used by the tokenizer.
    scaler : torch.cuda.amp.GradScaler, optional
        Gradient scaler for mixed precision training.

    Returns
    -------
    results: dict
        A dictionary containing the training performance for this epoch.
    total_step : int
        The total number of steps taken after this epoch.
    n_samples_seen : int
        The total number of samples seen after this epoch.
    """
    # Put the model in train mode
    model.train()

    if config.log_wandb:
        # Lazy import of wandb, since logging to wandb is optional
        import wandb

    loss_epoch = 0
    acc_epoch = 0
    acc_kpt_epoch = 0
    acc_all_epoch = 0

    if config.print_interval is None:
        # Default to printing to console every time we log to wandb
        config.print_interval = config.log_interval

    t_end_batch = time.time()
    t_start_wandb = t_end_wandb = None
    for batch_idx, (sequences, y_true, att_mask) in enumerate(dataloader):
        t_start_batch = time.time()
        batch_size_this_gpu = sequences.shape[0]

        # Move training inputs and targets to the GPU
        sequences = sequences.to(device)
        att_mask = att_mask.to(device)

        # Build the masking on the fly ----------------------------------------
        # t_start_masking = time.time()

        # Create a mask for allowed tokens i.e. that excludes all special tokens [<MASK>, <UNK>]
        special_tokens_mask = sequences > (n_special_tokens - 1)

        if config.tokenize_n_nucleotide:
            # Either exlude the last token [N..N] if config.predict_n_nucleotide == True
            # Or exclude all tokens containing Ns i.e "bad kamers" whose index in the vocab
            # is greater than 4**k
            special_tokens_mask &= sequences < (n_special_tokens + 4**config.k_mer - 1)

        special_tokens_mask = special_tokens_mask.to(device)
        masked_input = sequences.clone()
        random_mask = torch.rand(sequences.shape, device=device)
        masked_input = sequences.clone()
        random_mask = torch.rand(masked_input.shape, device=device)  # I can only do this for non-overlapping
        input_maskout = random_mask < mask_ratio
        input_maskout &= special_tokens_mask  # Cannot mask the special tokens including [UNK]
        masked_input[input_maskout] = 0

        # Forward pass --------------------------------------------------------
        t_start_forward = time.time()
        # N.B. To accurately time steps on GPU we need to use torch.cuda.Event
        ct_forward = torch.cuda.Event(enable_timing=True)
        ct_forward.record()
        
        # Perform the forward pass through the model with mixed precision
        if scaler is not None:
            from torch.cuda.amp import autocast
            with autocast():
                out = model(masked_input, attention_mask=att_mask)
                targets = sequences - n_special_tokens * (sequences > (n_special_tokens - 1))
                # Measure loss
                loss = criterion(
                    out.logits.view(-1, 4**config.k_mer)[special_tokens_mask.view(-1)],
                    targets.view(-1)[special_tokens_mask.view(-1)],
                )
        else:
            out = model(masked_input, attention_mask=att_mask)
            targets = sequences - n_special_tokens * (sequences > (n_special_tokens - 1))
            # Measure loss
            loss = criterion(
                out.logits.view(-1, 4**config.k_mer)[special_tokens_mask.view(-1)],
                targets.view(-1)[special_tokens_mask.view(-1)],
            )

        # Backward pass -------------------------------------------------------
        # Reset gradients
        optimizer.zero_grad()
        # Now the backward pass
        ct_backward = torch.cuda.Event(enable_timing=True)
        ct_backward.record()
        
        if scaler is not None:
            # Mixed precision backward pass
            scaler.scale(loss).backward()
        else:
            # Standard backward pass
            loss.backward()

        # Update --------------------------------------------------------------
        # Use our optimizer to update the model parameters
        ct_optimizer = torch.cuda.Event(enable_timing=True)
        ct_optimizer.record()
        
        if scaler is not None:
            # Mixed precision optimizer step
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard optimizer step
            optimizer.step()

        # Step the scheduler each batch
        scheduler.step()

        # Increment training progress counters
        total_step += 1
        batch_size_all = batch_size_this_gpu * config.world_size
        n_samples_seen += batch_size_all

        # Logging -------------------------------------------------------------
        # Log details about training progress
        t_start_logging = time.time()
        ct_logging = torch.cuda.Event(enable_timing=True)
        ct_logging.record()

        # Update the total loss for the epoch
        loss_batch = loss.clone()
        if config.distributed:
            # Fetch results from other GPUs
            dist.reduce(loss_batch, 0, op=dist.ReduceOp.AVG)
        loss_batch = loss_batch.item()
        loss_epoch += loss_batch

        with torch.no_grad():
            x_pred = torch.argmax(out.logits, dim=-1)

        if epoch <= 1 and batch_idx == 0:
            # Debugging
            print("sequences.shape     =", sequences.shape)
            print("y_true.shape        =", y_true.shape)
            print("input_maskout.shape =", input_maskout.shape)
            print("masked_input.shape  =", masked_input.shape)
            print("targets.shape       =", targets.shape)
            print("x_pred.shape        =", x_pred.shape)
            print("logits.shape        =", out.logits.shape)
            print("loss.shape          =", loss.shape)
            # Debugging intensifies
            print("sequences[0]     =", sequences[0])
            print("input_maskout[0] =", input_maskout[0])
            print("y_true[0]        =", y_true[0])
            print("masked_input[0]  =", masked_input[0])
            print("targets[0]       =", targets[0])
            print("x_pred[0]        =", x_pred[0])
            print("logits[0]        =", out.logits[0])
            print("loss =", loss.detach().item())

        # Compute accuracy
        with torch.no_grad():
            is_correct = x_pred == targets
            # Overall accuracy, including tokens which weren't masked out
            acc_all = is_correct[special_tokens_mask].sum() / is_correct[special_tokens_mask].numel()
            # Accuracy only on the masked tokens
            acc_msk = (
                is_correct[input_maskout & special_tokens_mask].sum() / (input_maskout & special_tokens_mask).sum()
            )
            # Accuracy only on the non-masked tokens
            acc_kpt = (
                is_correct[~input_maskout & special_tokens_mask].sum() / (~input_maskout & special_tokens_mask).sum()
            )
            if config.distributed:
                # Fetch results from other GPUs
                dist.reduce(acc_all, 0, op=dist.ReduceOp.AVG)
                dist.reduce(acc_msk, 0, op=dist.ReduceOp.AVG)
                dist.reduce(acc_kpt, 0, op=dist.ReduceOp.AVG)
            acc_all = 100.0 * acc_all.item()
            acc_msk = 100.0 * acc_msk.item()
            acc_kpt = 100.0 * acc_kpt.item()
            acc_epoch += acc_msk
            acc_kpt_epoch += acc_kpt
            acc_all_epoch += acc_all

        # Log to console
        if batch_idx <= 2 or batch_idx % config.print_interval == 0 or batch_idx >= len(dataloader) - 1:
            print(
                f"Train Epoch:{epoch:3d}" + (f"/{n_epoch}" if n_epoch is not None else ""),
                " Step:{:6d}/{}".format(batch_idx + 1, len(dataloader)),
                " Loss:{:8.5f}".format(loss_batch),
                " AccMask:{:6.2f}%".format(acc_msk),
                " AccSeen:{:6.2f}%".format(acc_kpt),
                " LR: {}".format(scheduler.get_last_lr()),
                flush=True,
            )

        # Compute mask ratio actually used
        mask_ratio_actual = input_maskout.sum() / input_maskout.nelement()
        if config.distributed:
            dist.reduce(mask_ratio_actual, 0, op=dist.ReduceOp.AVG)

        # Log to wandb
        if config.log_wandb and config.global_rank == 0 and batch_idx % config.log_interval == 0:
            # Create a log dictionary to send to wandb
            # Epoch progress interpolates smoothly between epochs
            epoch_progress = epoch - 1 + (batch_idx + 1) / len(dataloader)
            # Throughput is the number of samples processed per second
            throughput = batch_size_all / (t_start_logging - t_end_batch)
            log_dict = {
                "Pretraining/stepwise/epoch": epoch,
                "Pretraining/stepwise/epoch_progress": epoch_progress,
                "Pretraining/stepwise/n_samples_seen": n_samples_seen,
                "Pretraining/stepwise/Train/throughput": throughput,
                "Pretraining/stepwise/Train/loss": loss_batch,
                "Pretraining/stepwise/Train/accuracy": acc_msk,
                "Pretraining/stepwise/Train/accuracy_unmasked": acc_kpt,
                "Pretraining/stepwise/Train/accuracy_overall": acc_all,
                "Pretraining/stepwise/Train/mask_ratio": mask_ratio,
                "Pretraining/stepwise/Train/mask_ratio_actual": mask_ratio_actual,
            }
            # Track the learning rate of each parameter group
            for lr_idx in range(len(optimizer.param_groups)):
                if "name" in optimizer.param_groups[lr_idx]:
                    grp_name = optimizer.param_groups[lr_idx]["name"]
                elif len(optimizer.param_groups) == 1:
                    grp_name = ""
                else:
                    grp_name = f"grp{lr_idx}"
                if grp_name != "":
                    grp_name = f"-{grp_name}"
                grp_lr = optimizer.param_groups[lr_idx]["lr"]
                log_dict[f"Pretraining/stepwise/lr{grp_name}"] = grp_lr
            # Synchronize ensures everything has finished running on each GPU
            torch.cuda.synchronize()
            # Record how long it took to do each step in the pipeline
            if t_start_wandb is not None:
                # Record how long it took to send to wandb last time
                log_dict["Pretraining/stepwise/duration/wandb"] = t_end_wandb - t_start_wandb
            log_dict["Pretraining/stepwise/duration/dataloader"] = t_start_batch - t_end_batch
            log_dict["Pretraining/stepwise/duration/preamble"] = t_start_forward - t_start_batch
            log_dict["Pretraining/stepwise/duration/forward"] = ct_forward.elapsed_time(ct_backward) / 1000
            log_dict["Pretraining/stepwise/duration/backward"] = ct_backward.elapsed_time(ct_optimizer) / 1000
            log_dict["Pretraining/stepwise/duration/optimizer"] = ct_optimizer.elapsed_time(ct_logging) / 1000
            log_dict["Pretraining/stepwise/duration/overall"] = time.time() - t_end_batch
            t_start_wandb = time.time()
            log_dict["Pretraining/stepwise/duration/logging"] = t_start_wandb - t_start_logging
            # Send to wandb
            wandb.log(log_dict, step=total_step)
            t_end_wandb = time.time()

        # Record the time when we finished this batch
        t_end_batch = time.time()

    results = {
        "loss": loss_epoch / (batch_idx + 1),
        "accuracy": acc_epoch / (batch_idx + 1),
        "accuracy_unmasked": acc_kpt_epoch / (batch_idx + 1),
        "accuracy_overall": acc_all_epoch / (batch_idx + 1),
    }
    return results, total_step, n_samples_seen


def evaluate(
    config,
    model,
    criterion,
    dataloader,
    mask_ratio=0.5,
    device="cuda",
    distance_table=None,
    n_special_tokens=2,
):
    r"""
    Evaluate the encoder on the validation data.

    Parameters
    ----------
    config : argparse.Namespace or OmegaConf
        The global config object.
    model : torch.nn.Module
        The encoder/decoder network.
    criterion : torch.nn.Module
        The loss function.
    dataloader : torch.utils.data.DataLoader
        A dataloader for the training set.
    mask_ratio : float, default=0.5
        The ratio of tokens to mask out.
    device : str or torch.device, default="cuda"
        The device to use.

    Returns
    -------
    results: dict
        A dictionary containing the evaluation performance.
    """
    # Put the model in train mode
    model.eval()

    loss_epoch = 0
    acc_epoch = 0
    acc_kpt_epoch = 0
    acc_all_epoch = 0
    n_samples = 0

    if config.print_interval is None:
        # Default to printing to console every time we log to wandb
        config.print_interval = config.log_interval

    # Set the random seed to be stable for the evaluation
    # (This is stable if you change the batch size, but not if you change the number of GPU workers)
    rng = torch.Generator(device=device).manual_seed(config.global_rank)

    with torch.no_grad():
        for batch_idx, (sequences, _y_true, att_mask) in enumerate(dataloader):
            batch_size_this_gpu = sequences.shape[0]

            # Move training inputs and targets to the GPU
            sequences = sequences.to(device)
            att_mask = att_mask.to(device)

            # Build the masking on the fly ------------------------------------
            # t_start_masking = time.time()
            # Create a mask for allowed tokens i.e. that excludes all special tokens [<MASK>, <UNK>]
            special_tokens_mask = sequences > (n_special_tokens - 1)

            if config.tokenize_n_nucleotide:
                # Either exlude the last token [N..N] if config.predict_n_nucleotide == True
                # Or exclude all tokens containing Ns i.e "bad kamers" whose index in the vocab
                # is greater than 4**k
                special_tokens_mask &= sequences < (n_special_tokens + 4**config.k_mer - 1)

            special_tokens_mask = special_tokens_mask.to(device)
            masked_input = sequences.clone()
            random_mask = torch.rand(masked_input.shape, generator=rng, device=device)
            input_maskout = random_mask < mask_ratio
            input_maskout &= special_tokens_mask  # Cannot mask the special tokens including [UNK]
            masked_input[input_maskout] = 0

            # Forward pass ----------------------------------------------------
            out = model(masked_input, attention_mask=att_mask)
            # Measure loss
            targets = sequences - n_special_tokens * (sequences > (n_special_tokens - 1))
            loss = criterion(
                out.logits.view(-1, 4**config.k_mer)[special_tokens_mask.view(-1)],
                targets.view(-1)[special_tokens_mask.view(-1)],
            )

            # Metrics ---------------------------------------------------------
            # Update the total loss for the epoch
            loss_batch = loss * batch_size_this_gpu
            if config.distributed:
                # Fetch results from other GPUs
                dist.reduce(loss_batch, 0, op=dist.ReduceOp.SUM)
            loss_batch = loss_batch.item()
            loss_epoch += loss_batch

            # Compute accuracy
            x_pred = torch.argmax(out.logits, dim=-1)
            # Create a mask to ignore all special tokens [<MASK>, <UNK>]
            special_tokens_mask = targets > (n_special_tokens - 1)

            is_correct = x_pred == targets
            # Overall accuracy, including tokens which weren't masked out
            acc_all = (
                batch_size_this_gpu * is_correct[special_tokens_mask].sum() / is_correct[special_tokens_mask].numel()
            )
            # Accuracy only on the masked tokens
            acc_msk = (
                batch_size_this_gpu
                * is_correct[input_maskout & special_tokens_mask].sum()
                / (input_maskout & special_tokens_mask).sum()
            )
            # Accuracy only on the non-masked tokens
            acc_kpt = (
                batch_size_this_gpu
                * is_correct[~input_maskout & special_tokens_mask].sum()
                / (~input_maskout & special_tokens_mask).sum()
            )
            if config.distributed:
                # Fetch results from other GPUs
                dist.reduce(acc_all, 0, op=dist.ReduceOp.SUM)
                dist.reduce(acc_msk, 0, op=dist.ReduceOp.SUM)
                dist.reduce(acc_kpt, 0, op=dist.ReduceOp.SUM)
            acc_epoch += acc_msk
            acc_kpt_epoch += acc_kpt
            acc_all_epoch += acc_all

            # Total batch size
            if config.distributed:
                batch_size = torch.tensor(batch_size_this_gpu, device=device)
                dist.reduce(batch_size, 0, op=dist.ReduceOp.SUM)
                batch_size = batch_size.item()
            else:
                batch_size = batch_size_this_gpu
            n_samples += batch_size

            if batch_idx % (config.print_interval * 10) == 0:
                print(
                    "Eval",
                    " Step:{:6d}/{}".format(batch_idx + 1, len(dataloader)),
                    " Loss:{:8.5f}".format(loss_batch / batch_size),
                    " AccMask:{:6.2f}%".format(100.0 * acc_msk / batch_size),
                    " AccSeen:{:6.2f}%".format(100.0 * acc_kpt / batch_size),
                    flush=True,
                )

    results = {
        "loss": loss_epoch / n_samples,
        "accuracy": 100.0 * acc_epoch / n_samples,
        "accuracy_unmasked": 100.0 * acc_kpt_epoch / n_samples,
        "accuracy_overall": 100.0 * acc_all_epoch / n_samples,
        "n_samples": n_samples,
    }
    return results


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
        description="Pretrain BarcodeBERT.",
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
        default="BIOSCAN-5M",
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
        "--tokenizer",
        type=str,
        default="kmer",
        help="Name of tokenizer to use for DNA sequences. Default: %(default)s",
    )
    group.add_argument(
        "--k-mer",
        "--k_mer",
        type=int,
        default=4,
        help="Size of k-mer to use for DNA tokenization. Default: %(default)s",
    )
    group.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Stride to use for DNA tokenization. Default: %(default)s",
    )
    group.add_argument(
        "--max-len",
        "--max_len",
        type=int,
        default=660,
        help="Maximum length of input sequences. Default: %(default)s",
    )
    # Architecture args -------------------------------------------------------
    group = parser.add_argument_group("Architecture")
    group.add_argument(
        "--model",
        "--encoder",
        "--arch",
        "--architecture",
        dest="arch",
        type=str,
        default="transformer",
        help="Name of model architecture. Default: %(default)s",
    )
    group.add_argument(
        "--n-layers",
        "--n_layers",
        type=int,
        default=12,
        help="Number of layers in the transformer. Default: %(default)s",
    )
    group.add_argument(
        "--n-heads",
        "--n_heads",
        type=int,
        default=12,
        help="Number of attention heads in the transformer. Default: %(default)s",
    )
    group.add_argument(
        "--fast-att",
        "--fast_att",
        "--fast-attention",
        dest="fast_attention",
        action="store_true",
        help="Use FastBertForTokenClassification with optimized attention implementation using F.scaled_dot_product_attention.",
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
        default=0.0,
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
    group.add_argument(
        "--mixed-precision",
        "--mixed_precision",
        "--amp",
        dest="mixed_precision",
        action="store_true",
        help="Enable Automatic Mixed Precision (AMP) training for faster training and reduced memory usage.",
    )
    parser.add_argument(
        "--tokenize-n-nucleotide",
        "--tokenize_n_nucleotide",
        "--tokenize-n",
        "--tokenize_n",
        dest="tokenize_n_nucleotide",
        action="store_true",
        help="Include N as a valid character in tokenization. If false, all kmers including Ns will be mappend to [UNK]",
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
        default="BarcodeBERT",
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

    # kNN Evaluation args -----------------------------------------------------
    group = parser.add_argument_group("kNN evaluation")
    group.add_argument(
        "--knn-eval-frequency",
        "--knn_eval_frequency",
        type=int,
        default=5,
        help="Run kNN evaluation every N epochs. Set to 0 to disable. Default: %(default)s",
    )
    group.add_argument(
        "--knn-taxon",
        "--knn_taxon",
        type=str,
        default="genus",
        help="Taxonomic level for kNN evaluation. Default: %(default)s",
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
    return run(config)


if __name__ == "__main__":
    cli()
