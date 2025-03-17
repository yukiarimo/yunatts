import torch
from . import utils

def load_or_download_config(locale, use_hf=True, config_path=None):
    if config_path is None:
        raise ValueError("config_path is required")
    return utils.get_hparams_from_file(config_path)

def load_or_download_model(locale, device, use_hf=True, ckpt_path=None):
    if ckpt_path is None:
        raise ValueError("ckpt_path is required")
    return torch.load(ckpt_path, map_location=device)