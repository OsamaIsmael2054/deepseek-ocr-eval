# DeepSeek OCR Evaluation

A Python package for evaluating DeepSeek OCR model on OCR datasets and images.

## Installation

### Prerequisites

- Python 3.13+
- CUDA-capable GPU
- PyTorch 2.6+

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

#### Advanced Options

```bash
# Use flash attention for faster inference
python evaluate.py --dataset "NAMAA-Space/QariOCR-v0.3-markdown-mixed-dataset" --use-flash-attention


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
