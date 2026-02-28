"""
Utility functions for device management and deterministic behavior.
"""

import os
import random
from typing import Optional, Union

import numpy as np
import torch


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get the appropriate device for computation.
    
    Args:
        device: Device specification. If None, auto-detect.
        
    Returns:
        torch.device: The device to use for computation.
    """
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
    # Make deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device_info() -> dict:
    """
    Get information about available devices.
    
    Returns:
        dict: Device information including CUDA, MPS, and CPU availability.
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "cpu_available": True,
    }
    
    if info["cuda_available"]:
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_current_device"] = torch.cuda.current_device()
        info["cuda_device_name"] = torch.cuda.get_device_name()
    
    return info


def move_to_device(
    obj: Union[torch.Tensor, dict, list, tuple], 
    device: torch.device
) -> Union[torch.Tensor, dict, list, tuple]:
    """
    Move object to specified device.
    
    Args:
        obj: Object to move (tensor, dict, list, or tuple).
        device: Target device.
        
    Returns:
        Object moved to the specified device.
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)([move_to_device(item, device) for item in obj])
    else:
        return obj
