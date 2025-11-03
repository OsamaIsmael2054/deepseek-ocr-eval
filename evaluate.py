import argparse
import sys
import os
import re
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import requests
from io import BytesIO
from datasets import load_dataset
from tqdm import tqdm
import jiwer

from deepseek_ocr_eval import infer_with_image_path, infer_with_image_object, extract_clean_text


def load_model(model_name: str = 'deepseek-ai/DeepSeek-OCR', use_flash_attention: bool = False):
    """
    Load the DeepSeek OCR model and tokenizer.
    
    Args:
        model_name: HuggingFace model name
    
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if use_flash_attention:
        model = AutoModel.from_pretrained(model_name, attn_implementation='flash_attention_2', trust_remote_code=True, use_safetensors=True, torch_dtype=torch.bfloat16, device_map='cuda:0')
    else:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True, use_safetensors=True, torch_dtype=torch.bfloat16, device_map='cuda:0')
    model = model.eval()
    print("Model loaded successfully!")
    return model, tokenizer


def load_image_from_url(url: str) -> Image.Image:
    """
    Load an image from a URL.
    
    Args:
        url: Image URL
    
    Returns:
        PIL Image object
    """
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")


def load_image_from_file(path: str) -> Image.Image:
    """
    Load an image from a file.
    
    Args:
        path: Image file path
    
    Returns:
        PIL Image object
    """
    return Image.open(path).convert("RGB")


def evaluate_single_image(
    model, 
    tokenizer, 
    image_path: str = None,
    image_url: str = None,
    prompt: str = '<image>\n<|grounding|>Convert the document to markdown. ',
    output_path: str = 'output',
    base_size: int = 1024,
    image_size: int = 640,
    crop_mode: bool = True,
    save_results: bool = False,
    eval_mode: bool = True
):
    """
    Evaluate the model on a single image.
    
    Args:
        model: DeepSeek OCR model
        tokenizer: Tokenizer
        image_path: Path to local image file
        image_url: URL to image
        prompt: Prompt for the model
        output_path: Directory to save outputs
        clean_text: Whether to clean the output text
    """
    # Load image
    if image_url:
        print(f"Loading image from URL: {image_url}")
        image = load_image_from_url(image_url)
        print("Infer Image from URL...")
        result = infer_with_image_object(
            model, 
            tokenizer, 
            image, 
            prompt=prompt,
            output_path=output_path,
            base_size=base_size,
            image_size=image_size,
            crop_mode=crop_mode,
            save_results=save_results,
            eval_mode=eval_mode
        )
        
    elif image_path:
        print(f"Loading image from file: {image_path}")
        image = load_image_from_file(image_path)
        print("Infer Image from file...")
        result = infer_with_image_path(
            model, 
            tokenizer, 
            image, 
            prompt=prompt,
            output_path=output_path,
            base_size=base_size,
            image_size=image_size,
            crop_mode=crop_mode,
            save_results=save_results,
            eval_mode=eval_mode
        )
    else:
        raise ValueError("Either image_path or image_url must be provided")
    
    return result


def evaluate_dataset(
    model,
    tokenizer,
    dataset_name: str = "NAMAA-Space/QariOCR-v0.3-markdown-mixed-dataset",
    dataset_split: str = "test",
    num_samples: int = None,
    prompt: str = '<image>\n<|grounding|>Convert the document to markdown. ',
    output_dir: str = 'output',
    base_size: int = 1024,
    image_size: int = 640,
    crop_mode: bool = False,
    save_results: bool = False,
    eval_mode: bool = True
):
    """
    Evaluate the model on a dataset.
    
    Args:
        model: DeepSeek OCR model
        tokenizer: Tokenizer
        dataset_name: HuggingFace dataset name
        dataset_split: Dataset split to use
        num_samples: Number of samples to evaluate (None = all)
        prompt: Prompt for the model
        output_dir: Base directory for outputs
        base_size: Base size for the model
        image_size: Image size for the model
        crop_mode: Crop mode for the model
        save_results: Whether to save results
        eval_mode: Evaluation mode for the model
    """
    print(f"Loading dataset: {dataset_name}, split: {dataset_split}")
    ds = load_dataset(dataset_name, split=dataset_split)
    
    if num_samples:
        ds = ds.select(range(min(num_samples, len(ds))))
    
    print(f"Evaluating {len(ds)} samples...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    wer_scores = []
    cer_scores = []
    
    # Open JSONL file for writing results
    jsonl_file = os.path.join(output_dir, 'results.jsonl')
    
    with open(jsonl_file, 'w', encoding='utf-8') as jsonl_f:
        for idx, sample in tqdm(enumerate(ds), total=len(ds), desc="Processing"):        
            # Load image
            image_url = sample['image']
            image = load_image_from_url(image_url)
            
            # Run inference
            result = infer_with_image_object(
                model, 
                tokenizer, 
                image, 
                prompt=prompt,
                output_path="",  # Don't save individual files
                base_size=base_size,
                image_size=image_size,
                crop_mode=crop_mode,
                save_results=False,  # Don't save intermediate results
                eval_mode=eval_mode
            )
            
            # Clean text
            result = extract_clean_text(result) if result else ""
            
            # Get ground truth
            ground_truth = sample.get('text', '')
            ground_truth = re.sub(r'<[^>]+>','',ground_truth)
            
            # Calculate WER and CER if ground truth is available
            sample_wer = None
            sample_cer = None
            if ground_truth and result:
                try:
                    # Calculate WER (Word Error Rate)
                    sample_wer = jiwer.wer(ground_truth, result)
                    wer_scores.append(sample_wer)
                    
                    # Calculate CER (Character Error Rate)
                    sample_cer = jiwer.cer(ground_truth, result)
                    cer_scores.append(sample_cer)
                except Exception as e:
                    print(f"Warning: Could not calculate metrics for sample {idx}: {e}")
            
            # Create JSON record
            record = {
                'idx': idx,
                'image_url': image_url,
                'reference': ground_truth,
                'prediction': result,
                'wer': sample_wer,
                'cer': sample_cer
            }
            
            # Write to JSONL file
            jsonl_f.write(json.dumps(record, ensure_ascii=False) + '\n')
            jsonl_f.flush()  # Ensure data is written immediately
            
            results.append(record)
            
            if sample_wer is not None and sample_cer is not None:
                print(f"Sample {idx + 1}/{len(ds)} - WER: {sample_wer:.4f}, CER: {sample_cer:.4f}")
    
    # Calculate average metrics
    avg_wer = sum(wer_scores) / len(wer_scores) if wer_scores else None
    avg_cer = sum(cer_scores) / len(cer_scores) if cer_scores else None
    
    # Save summary
    summary_file = os.path.join(output_dir, 'summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"Evaluation Summary\n")
        f.write(f"{'='*80}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Split: {dataset_split}\n")
        f.write(f"Samples evaluated: {len(results)}\n")
        f.write(f"Samples with metrics: {len(wer_scores)}\n")
        f.write(f"\n")
        if avg_wer is not None and avg_cer is not None:
            f.write(f"Average Metrics:\n")
            f.write(f"  Word Error Rate (WER): {avg_wer:.4f} ({avg_wer*100:.2f}%)\n")
            f.write(f"  Character Error Rate (CER): {avg_cer:.4f} ({avg_cer*100:.2f}%)\n")
        else:
            f.write(f"No metrics calculated (no ground truth available)\n")
        f.write(f"\n")
        f.write(f"Output Files:\n")
        f.write(f"  - results.jsonl: Full results with predictions, references, and metrics\n")
        f.write(f"  - metrics_summary.csv: Summary of metrics per sample\n")
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Evaluation Complete!")
    print(f"{'='*80}")
    print(f"Samples evaluated: {len(results)}")
    if avg_wer is not None and avg_cer is not None:
        print(f"Average WER: {avg_wer:.4f} ({avg_wer*100:.2f}%)")
        print(f"Average CER: {avg_cer:.4f} ({avg_cer*100:.2f}%)")
    print(f"\nOutput files:")
    print(f"  - JSONL results: {jsonl_file}")
    print(f"  - Summary: {summary_file}")
    print(f"{'='*80}\n")
    
    # Save detailed metrics as CSV
    import csv
    csv_file = os.path.join(output_dir, 'metrics_summary.csv')
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['idx', 'wer', 'cer', 'has_prediction', 'has_reference'])
        writer.writeheader()
        for result in results:
            writer.writerow({
                'idx': result['idx'],
                'wer': f"{result['wer']:.4f}" if result['wer'] is not None else 'N/A',
                'cer': f"{result['cer']:.4f}" if result['cer'] is not None else 'N/A',
                'has_prediction': bool(result['prediction']),
                'has_reference': bool(result['reference'])
            })
    print(f"  - Metrics CSV: {csv_file}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate DeepSeek OCR models on images or datasets'
    )
    
    # Model arguments
    parser.add_argument('--model', type=str, default='deepseek-ai/DeepSeek-OCR',
                       help='HuggingFace model name')
    
    # Image arguments
    parser.add_argument('--image-path', type=str, help='Path to local image file')
    parser.add_argument('--image-url', type=str, help='URL to image')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str,
                       help='HuggingFace dataset name')
    parser.add_argument('--dataset-split', type=str, default='test',
                       help='Dataset split to use (default: test)')
    parser.add_argument('--num-samples', type=int,
                       help='Number of samples to evaluate (default: all)')
    
    # Inference arguments
    parser.add_argument('--prompt', type=str,
                       default='<image>\n<|grounding|>Convert the document to markdown. ',
                       help='Prompt for the model')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Directory to save outputs (default: output)')
    parser.add_argument('--use-flash-attention', action='store_true',
                       help='Use flash attention', default=False)
    parser.add_argument('--base-size', type=int, default=1024,
                       help='Base size for the model')
    parser.add_argument('--image-size', type=int, default=640,
                       help='Image size for the model')
    parser.add_argument('--crop-mode', action='store_true',
                       help='Crop mode for the model', default=True)
    parser.add_argument('--save-results', action='store_true',
                       help='Save results', default=False)
    parser.add_argument('--eval-mode', action='store_true',
                       help='Evaluation mode for the model', default=True)

    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.image_path, args.image_url, args.dataset]):
        parser.error("Must provide either --image-path, --image-url, or --dataset")
    
    # Load model
    model, tokenizer = load_model(args.model, args.use_flash_attention)
    
    # Evaluate
    if args.dataset:
        evaluate_dataset(
            model,
            tokenizer,
            dataset_name=args.dataset,
            dataset_split=args.dataset_split,
            num_samples=args.num_samples,
            prompt=args.prompt,
            output_dir=args.output_dir,
            base_size = args.base_size,
            image_size = args.image_size,
            crop_mode = args.crop_mode,
            save_results = args.save_results,
            eval_mode = args.eval_mode
        )
    else:
        evaluate_single_image(
            model,
            tokenizer,
            image_path=args.image_path,
            image_url=args.image_url,
            prompt=args.prompt,
            output_path=args.output_dir,
            base_size = args.base_size,
            image_size = args.image_size,
            crop_mode = args.crop_mode,
            save_results = args.save_results,
            eval_mode = args.eval_mode
        )


if __name__ == '__main__':
    main()

