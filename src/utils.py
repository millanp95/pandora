import os
import torch
import warnings

from .models import CNN_MLM


def remove_extra_pre_fix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        new_state_dict[key] = value
    return new_state_dict


def safe_save_model(modules, checkpoint_path=None, config=None, **kwargs):
    """
    Save a model to a checkpoint file, along with any additional data.

    Parameters
    ----------
    modules : dict
        A dictionary of modules to save. The keys are the names of the modules
        and the values are the modules themselves.
    checkpoint_path : str, optional
        Path to the checkpoint file. If not provided, the path will be taken
        from the config object.
    config : :class:`argparse.Namespace`, optional
        A configuration object containing the checkpoint path.
    **kwargs
        Additional data to save to the checkpoint file.
    """

    # 1) Determine checkpoint_path
    if checkpoint_path is None:
        if config is not None and hasattr(config, "checkpoint_path"):
            checkpoint_path = config.checkpoint_path
        else:
            raise ValueError("No checkpoint_path provided")
    # 2) Make sure output dir exists
    folder, fname = os.path.split(checkpoint_path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    # 3) Build temp filename
    tmp_fname = os.path.join(folder or ".", f".tmp.{fname}")
    print(tmp_fname)
    
    # 4) Assemble state dict
    data = {name: module.state_dict() for name, module in modules.items()}
    data.update(kwargs)
    if config is not None:
        data["config"] = config

    # 5) Save + atomic replace
    torch.save(data, tmp_fname)
    os.replace(tmp_fname, checkpoint_path)  # atomic on POSIX
    print(f"Saved model to {checkpoint_path}")


def load_pretrained_model(checkpoint_path, device=None):
    """
    Load a pretrained model from a checkpoint file.

    Parameters
    ----------
    checkpoint_path : str
        Path to the pretrained checkpoint file.

    Returns
    -------
    model : torch.nn.Module
        The pretrained model.
    ckpt : dict
        The contents of the checkpoint file.
    """
    print(f"\nLoading model from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = CNN_MLM(ckpt["config"].max_len)
    model.load_state_dict(remove_extra_pre_fix(ckpt["model"]))
    model.eval()
    print(
        f"Loaded model from {checkpoint_path}, resuming trainig from epoch {ckpt['epoch']}"
    )
    return model, ckpt


def check_is_distributed():
    r"""
    Check if the current job is running in distributed mode.

    Returns
    -------
    bool
        Whether the job is running in distributed mode.
    """
    return (
        "WORLD_SIZE" in os.environ
        and "RANK" in os.environ
        and "LOCAL_RANK" in os.environ
        and "MASTER_ADDR" in os.environ
        and "MASTER_PORT" in os.environ
    )


def setup_slurm_distributed():
    r"""
    Use SLURM environment variables to set up environment variables needed for DDP.

    Note: This is not used when using torchrun, as that sets RANK etc. for us,
    but is useful if you're using srun without torchrun (i.e. using srun within
    the sbatch file to lauching one task per GPU).
    """
    if "WORLD_SIZE" in os.environ:
        pass
    elif "SLURM_NNODES" in os.environ and "SLURM_GPUS_ON_NODE" in os.environ:
        os.environ["WORLD_SIZE"] = str(
            int(os.environ["SLURM_NNODES"]) * int(os.environ["SLURM_GPUS_ON_NODE"])
        )
    elif "SLURM_NPROCS" in os.environ:
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
    if "RANK" not in os.environ and "SLURM_PROCID" in os.environ:
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
        if int(os.environ["RANK"]) > 0 and "WORLD_SIZE" not in os.environ:
            raise EnvironmentError(
                f"SLURM_PROCID is {os.environ['SLURM_PROCID']}, implying"
                " distributed training, but WORLD_SIZE could not be determined."
            )
    if "LOCAL_RANK" not in os.environ and "SLURM_LOCALID" in os.environ:
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
    if "MASTER_ADDR" not in os.environ and "SLURM_NODELIST" in os.environ:
        os.environ["MASTER_ADDR"] = os.environ["SLURM_NODELIST"].split("-")[0]
    if "MASTER_PORT" not in os.environ and "SLURM_JOB_ID" in os.environ:
        os.environ["MASTER_PORT"] = str(49152 + int(os.environ["SLURM_JOB_ID"]) % 16384)


def get_num_cpu_available():
    r"""
    Get the number of available CPU cores.

    Uses :func:`os.sched_getaffinity` if available, otherwise falls back to
    :func:`os.cpu_count`.

    Returns
    -------
    ncpus : int
        The number of available CPU cores.
    """
    try:
        # This is the number of CPUs available to this process, which may
        # be smaller than the number of CPUs on the system.
        # This command is only available on Unix-like systems.
        return len(os.sched_getaffinity(0))
    except Exception:
        # Fall-back for Windows or other systems which don't support sched_getaffinity
        warnings.warn(
            "Unable to determine number of available CPUs available to this python"
            " process specifically. Falling back to the total number of CPUs on the"
            " system.",
            RuntimeWarning,
            stacklevel=2,
        )
        return os.cpu_count()
