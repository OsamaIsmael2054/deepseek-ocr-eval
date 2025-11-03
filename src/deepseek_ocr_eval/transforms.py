from typing import Optional, Tuple
import torch
import torch.nn as nn
from torchvision import transforms


def normalize_transform(mean, std):
    """
    Create a normalization transform.
    
    Args:
        mean: Mean values for normalization
        std: Standard deviation values for normalization
    
    Returns:
        Normalization transform or None
    """
    if mean is None and std is None:
        transform = None
    elif mean is None and std is not None:
        mean = [0.] * len(std)
        transform = transforms.Normalize(mean=mean, std=std)
    elif mean is not None and std is None:
        std = [1.] * len(mean)
        transform = transforms.Normalize(mean=mean, std=std)
    else:
        transform = transforms.Normalize(mean=mean, std=std)

    return transform


class BasicImageTransform:
    """
    Basic image transformation pipeline.
    
    Converts PIL images to tensors and applies normalization.
    """
    
    def __init__(
        self, 
        mean: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
        std: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
        normalize: bool = True
    ):
        """
        Initialize the image transform.
        
        Args:
            mean: Mean values for normalization
            std: Standard deviation values for normalization
            normalize: Whether to apply normalization
        """
        self.mean = mean
        self.std = std
    
        transform_pipelines = [
            transforms.ToTensor()
        ]

        normalize_fn = normalize_transform(mean, std) if normalize else nn.Identity()
        if normalize_fn is not None:
            transform_pipelines.append(normalize_fn)

        self.transform = transforms.Compose(transform_pipelines)
    
    def __call__(self, x):
        """Apply the transform to an image."""
        x = self.transform(x)
        return x

