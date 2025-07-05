import os
import torch
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
    if checkpoint_path is not None:
        pass
    elif config is not None and hasattr(config, "checkpoint_path"):
        checkpoint_path = config.checkpoint_path
    else:
        raise ValueError("No checkpoint path provided")
    print(f"\nSaving model to {checkpoint_path}")
    # Save to a temporary file first, then move the temporary file to the target
    # destination. This is to prevent clobbering the checkpoint with a partially
    # saved file, in the event that the saving process is interrupted. Saving
    # the checkpoint takes a little while and can be disrupted by preemption,
    # whereas moving the file is an atomic operation.
    tmp_a, tmp_b = os.path.split(checkpoint_path)
    tmp_fname = os.path.join(tmp_a, ".tmp." + tmp_b)
    data = {k: v.state_dict() for k, v in modules.items()}
    data.update(kwargs)
    if config is not None:
        data["config"] = config

    torch.save(data, tmp_fname)
    os.rename(tmp_fname, checkpoint_path)
    print(f"Saved model to  {checkpoint_path}")


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

    assert "bert_config" in ckpt  # You may be trying to load an old checkpoint

    model = CNN_MLM()
    model.load_state_dict(remove_extra_pre_fix(ckpt["model"]))
    model.eval()
    print(f"Loaded model from {checkpoint_path}")
    return model, ckpt
