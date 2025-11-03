# DeepSeek OCR Evaluation

A Python package for evaluating DeepSeek OCR models on OCR datasets and images.

## Features

- 🚀 Easy-to-use CLI for OCR evaluation
- 📊 Support for HuggingFace datasets
- 🖼️ Process single images or entire datasets
- 🔧 Flexible prompt customization
- 📝 Automatic text cleaning and post-processing
- 💾 JSONL output format for easy analysis
- 📈 WER and CER metrics using jiwer
- ⚡ Flash Attention support for faster inference
- 🎯 Configurable image processing (base size, crop mode)

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- PyTorch 2.0+

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

The main entry point is `evaluate.py` which provides a comprehensive CLI for OCR evaluation.

#### Evaluate a Single Image from URL

```bash
python evaluate.py --image-url "https://example.com/image.jpg"
```

#### Evaluate a Local Image File

```bash
python evaluate.py --image-path "path/to/image.jpg"
```

#### Evaluate a Dataset

```bash
python evaluate.py --dataset "NAMAA-Space/QariOCR-v0.3-markdown-mixed-dataset" --num-samples 10
```

#### Custom Prompt

```bash
python evaluate.py --image-path "image.jpg" --prompt "<image>\nExtract all text from this document."
```

#### Advanced Options

```bash
# Use flash attention for faster inference
python evaluate.py --dataset "NAMAA-Space/QariOCR-v0.3-markdown-mixed-dataset" --use-flash-attention

# Custom image processing parameters
python evaluate.py --dataset "NAMAA-Space/QariOCR-v0.3-markdown-mixed-dataset" \
  --base-size 1280 \
  --image-size 640 \
  --crop-mode

# With all options
python evaluate.py \
  --dataset "NAMAA-Space/QariOCR-v0.3-markdown-mixed-dataset" \
  --num-samples 100 \
  --use-flash-attention \
  --base-size 1024 \
  --image-size 640 \
  --crop-mode \
  --eval-mode \
  --output-dir results
```

#### Full Options

```bash
python evaluate.py --help
```

Available options:

**Model & Input:**
- `--model`: HuggingFace model name (default: `deepseek-ai/DeepSeek-OCR`)
- `--image-path`: Path to local image file
- `--image-url`: URL to image
- `--dataset`: HuggingFace dataset name
- `--dataset-split`: Dataset split to use (default: `test`)
- `--num-samples`: Number of samples to evaluate (default: all)

**Prompt & Output:**
- `--prompt`: Custom prompt for the model (default: `<image>\n<|grounding|>Convert the document to markdown.`)
- `--output-dir`: Directory to save outputs (default: `output`)

**Model Configuration:**
- `--use-flash-attention`: Enable Flash Attention 2 for faster inference
- `--base-size`: Base image size for processing (default: `1024`)
- `--image-size`: Crop image size (default: `640`)
- `--crop-mode`: Enable dynamic image cropping
- `--eval-mode`: Enable evaluation mode (no streaming)
- `--save-results`: Save intermediate results (boxes, images, etc.)

### Python API

You can also use the package programmatically:

```python
import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
from deepseek_ocr_eval import infer_with_image_object, extract_clean_text

# Load model
model_name = 'deepseek-ai/DeepSeek-OCR'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True, use_safetensors=True)
model = model.eval().cuda().to(torch.bfloat16)

# Load image
image = Image.open("path/to/image.jpg").convert("RGB")

# Run inference
result = infer_with_image_object(
    model,
    tokenizer,
    image,
    prompt='<image>\n<|grounding|>Convert the document to markdown. ',
    output_path='output',
    eval_mode=True
)

# Clean output
clean_result = extract_clean_text(result)
print(clean_result)
```

## Project Structure

```
deepseek-ocr-eval/
├── src/
│   └── deepseek_ocr_eval/
│       ├── __init__.py          # Package initialization
│       ├── conversation.py      # Conversation templates
│       ├── transforms.py        # Image transformations
│       ├── inference.py         # Model inference logic
│       └── utils.py            # Utility functions
├── evaluate.py                  # Main CLI script
├── requirements.txt             # Dependencies
├── README.md                    # This file
└── notebookad01e83a17.ipynb    # Original notebook
```

## Module Overview

### `conversation.py`

Manages conversation templates and prompt formatting for different model variants:
- `Conversation`: Main conversation class
- `SeparatorStyle`: Enum for different separator styles
- `format_messages()`: Format conversations into prompts

### `transforms.py`

Image preprocessing and transformations:
- `BasicImageTransform`: Standard image transform pipeline
- `normalize_transform()`: Create normalization transforms

### `inference.py`

Core inference functionality:
- `infer_with_image_object()`: Main inference function
- `NoEOSTextStreamer`: Custom text streamer for generation

### `utils.py`

Utility functions:
- `dynamic_preprocess()`: Dynamic image cropping and preprocessing
- `text_encode()`: Text tokenization helper
- `extract_clean_text()`: Clean model outputs

## Advanced Usage

### Custom Image Processing

```python
from deepseek_ocr_eval import infer_with_image_object

result = infer_with_image_object(
    model,
    tokenizer,
    image,
    prompt='<image>\nExtract all text.',
    base_size=1024,        # Base image size
    image_size=640,        # Crop size
    crop_mode=True,        # Enable dynamic cropping
    eval_mode=True
)
```

### Batch Processing

```python
from pathlib import Path
from PIL import Image

image_dir = Path("images/")
results = []

for image_path in image_dir.glob("*.jpg"):
    image = Image.open(image_path).convert("RGB")
    result = infer_with_image_object(model, tokenizer, image)
    results.append({
        'filename': image_path.name,
        'prediction': result
    })
```

## Output Structure

When evaluating datasets, the output directory will be organized as follows:

```
output/
├── results.jsonl           # All results in JSONL format
├── metrics_summary.csv     # Per-sample metrics (WER, CER)
└── summary.txt            # Evaluation summary with avg metrics
```

### JSONL Format

Each line in `results.jsonl` contains a JSON object:

```json
{
  "idx": 0,
  "image_url": "https://example.com/image.jpg",
  "reference": "ground truth text here",
  "prediction": "model prediction here",
  "wer": 0.1234,
  "cer": 0.0567
}
```

### Reading Results

```python
import json

# Read JSONL results
with open('output/results.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        record = json.loads(line)
        print(f"Sample {record['idx']}: WER={record['wer']:.4f}, CER={record['cer']:.4f}")
```

Or with pandas:

```python
import pandas as pd
import json

results = []
with open('output/results.jsonl', 'r') as f:
    for line in f:
        results.append(json.loads(line))

df = pd.DataFrame(results)
print(df[['idx', 'wer', 'cer']].describe())
```

## Requirements

Key dependencies:
- `torch>=2.0.0`: PyTorch for model inference
- `transformers==4.46.3`: HuggingFace Transformers
- `tokenizers==0.20.3`: Fast tokenizers
- `Pillow>=9.0.0`: Image processing
- `datasets==4.3.0`: HuggingFace datasets
- `einops==0.8.1`: Tensor operations
- `jiwer>=4.0.0`: WER and CER metrics

See `requirements.txt` for the complete list.

## Evaluation Metrics

The evaluation automatically calculates:

- **WER (Word Error Rate)**: Measures word-level accuracy
  - Formula: `(Substitutions + Deletions + Insertions) / Total Words`
  - Range: 0.0 (perfect) to ∞
  - Lower is better

- **CER (Character Error Rate)**: Measures character-level accuracy
  - Formula: `(Substitutions + Deletions + Insertions) / Total Characters`
  - Range: 0.0 (perfect) to ∞
  - Lower is better

Metrics are automatically calculated when ground truth is available in the dataset.

## Notes

- The model requires a CUDA-capable GPU for optimal performance
- Default model uses `torch.bfloat16` precision
- First run will download the model from HuggingFace (~several GB)
- Some models require accepting license agreements on HuggingFace
- Flash Attention 2 requires compatible GPU (Ampere or newer) and proper installation
- Text cleaning is automatically applied to all predictions

## Citation

If you use DeepSeek OCR in your research, please cite:

```bibtex
@software{deepseek_ocr,
  title = {DeepSeek OCR},
  author = {DeepSeek AI},
  year = {2024},
  url = {https://huggingface.co/deepseek-ai/DeepSeek-OCR}
}
```

## License

This project is provided as-is. Please refer to the DeepSeek OCR model license for model usage terms.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Troubleshooting

### CUDA Out of Memory

If you encounter CUDA OOM errors:
- Reduce `--base-size` and `--image-size` parameters
- Process images in smaller batches with `--num-samples`
- Disable `--crop-mode` to reduce memory usage
- Use a GPU with more VRAM
- Try using `--use-flash-attention` for more efficient memory usage

### Model Download Issues

If model download fails:
- Check your internet connection
- Verify HuggingFace credentials (some models require authentication)
- Use `huggingface-cli login` to authenticate

### Import Errors

Make sure all dependencies are installed:
```bash
pip install -r requirements.txt --upgrade
```

### Flash Attention Issues

If `--use-flash-attention` fails:
- Ensure you have a compatible GPU (Ampere or newer: RTX 3000+, A100, H100)
- Install flash-attention: `pip install flash-attn --no-build-isolation`
- Remove the flag to use standard attention

### WER/CER Calculation Errors

If metrics calculation fails:
- Ensure ground truth text is available in the dataset
- Check that both prediction and reference texts are valid strings
- Metrics will be skipped automatically if ground truth is missing

## Example Output

When running evaluation on a dataset:

```bash
$ python evaluate.py --dataset "NAMAA-Space/QariOCR-v0.3-markdown-mixed-dataset" --num-samples 10

Loading model: deepseek-ai/DeepSeek-OCR
Model loaded successfully!
Loading dataset: NAMAA-Space/QariOCR-v0.3-markdown-mixed-dataset, split: test
Evaluating 10 samples...
Processing: 100%|████████████████████| 10/10 [02:30<00:00, 15.0s/it]
Sample 1/10 - WER: 0.1234, CER: 0.0567
Sample 2/10 - WER: 0.0987, CER: 0.0423
...

================================================================================
Evaluation Complete!
================================================================================
Samples evaluated: 10
Average WER: 0.1111 (11.11%)
Average CER: 0.0495 (4.95%)

Output files:
  - JSONL results: output/results.jsonl
  - Summary: output/summary.txt
  - Metrics CSV: output/metrics_summary.csv
================================================================================
```

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.

