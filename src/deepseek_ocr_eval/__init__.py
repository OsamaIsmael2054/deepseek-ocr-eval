"""
DeepSeek OCR Evaluation Package

A package for evaluating DeepSeek OCR models on OCR datasets.
"""

__version__ = "0.1.0"

from .conversation import (
    Conversation,
    SeparatorStyle,
    get_conv_template,
    format_messages
)
from .transforms import BasicImageTransform, normalize_transform
from .inference import infer_with_image_object, infer_with_image_path
from .utils import (
    dynamic_preprocess,
    text_encode,
    extract_clean_text
)

__all__ = [
    "Conversation",
    "SeparatorStyle",
    "get_conv_template",
    "format_messages",
    "BasicImageTransform",
    "normalize_transform",
    "infer_with_image_object",
    "infer_with_image_path",
    "dynamic_preprocess",
    "text_encode",
    "extract_clean_text"
]

