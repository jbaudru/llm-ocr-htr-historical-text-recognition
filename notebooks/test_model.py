import os
import torch
import pandas as pd
import random
import csv
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
from torchmetrics.text import CharErrorRate

# Set the model paths - try both base model and finetuned model
BASE_MODEL = "naver-clova-ix/donut-base"
FINETUNED_MODEL = os.path.join(os.getcwd(), "models", "finetuned_donut_vlm")
DATASET_CSV = "handwritten_dataset.csv"  # Path to the dataset CSV with ground truth

# Check if CUDA is available and fallback to CPU if not
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def load_model(model_path_or_name):
    """Load the processor and model from the specified path or name"""
    print(f"Loading model from: {model_path_or_name}")
    try:
        processor = DonutProcessor.from_pretrained(model_path_or_name)
        model = VisionEncoderDecoderModel.from_pretrained(model_path_or_name)
        
        # Configure image processor for better document handling
        processor.image_processor.size = {"height": 480, "width": 960}
        
        # Configure special tokens if they're not already in the tokenizer
        special_tokens = ["<s_text>", "</s_text>"]
        if special_tokens[0] not in processor.tokenizer.get_vocab():
            print("Adding special tokens to tokenizer")
            processor.tokenizer.add_tokens(special_tokens)
            model.decoder.resize_token_embeddings(len(processor.tokenizer))
        
        # Set up model configuration for generation
        model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(["<s_text>"])[0]
        model.config.pad_token_id = processor.tokenizer.pad_token_id
        model.config.vocab_size = len(processor.tokenizer)
        
        # Set special generation parameters for better OCR results
        model.config.decoder.min_length = 1 
        model.config.decoder.max_length = 128
        model.config.decoder.no_repeat_ngram_size = 3
        model.config.decoder.early_stopping = True
        model.config.decoder.num_beams = 4
        
        # Move model to appropriate device
        model.to(device)
        
        return processor, model
    except Exception as e:
        print(f"Error loading model from {model_path_or_name}: {e}")
        return None, None

def find_image_path(base_filename, is_line=False):
    """Find the correct path for an image by trying multiple locations"""
    # Determine the subdirectory based on whether we're looking for line or page images
    img_subdir = "lines" if is_line else "full_pages"
    
    # List of possible paths to try
    possible_paths = [
        # Current directory path
        os.path.join(os.getcwd(), "lam_data", "archive", "LAM", img_subdir, "img", base_filename),
        
        # Full path
        os.path.join(f"C:\\Users\\Administrateur\\Documents\\GitHub\\img-analysis_seorin_project\\lam_data\\archive\\LAM\\{img_subdir}\\img", base_filename),
        
        # Path relative to notebooks directory
        os.path.join(os.getcwd(), "..", "lam_data", "archive", "LAM", img_subdir, "img", base_filename),
        
        # Try in the repo root directory
        os.path.join(os.path.dirname(os.getcwd()), "lam_data", "archive", "LAM", img_subdir, "img", base_filename)
    ]
    
    # Try each path
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    print(f"Could not find image: {base_filename}")
    return None

def load_ground_truth_data():
    """Load the ground truth data from the handwritten_dataset.csv file"""
    try:
        # Try different possible locations for the CSV
        possible_csv_paths = [
            DATASET_CSV,
            os.path.join(os.getcwd(), DATASET_CSV),
            os.path.join(os.getcwd(), "..", DATASET_CSV),
            os.path.join(os.path.dirname(os.getcwd()), DATASET_CSV)
        ]
        
        for csv_path in possible_csv_paths:
            if os.path.exists(csv_path):
                print(f"Loading ground truth data from: {csv_path}")
                df = pd.read_csv(csv_path)
                # Create a dictionary mapping image filenames to their ground truth text
                ground_truth_dict = {}
                for _, row in df.iterrows():
                    image_filename = os.path.basename(row['image_path'])
                    ground_truth_dict[image_filename] = row['text']
                    # Also add the full path as a key
                    ground_truth_dict[row['image_path']] = row['text']
                
                print(f"Loaded {len(ground_truth_dict) // 2} ground truth entries")
                return ground_truth_dict, df
        
        print(f"Could not find {DATASET_CSV} file!")
        return {}, None
    except Exception as e:
        print(f"Error loading ground truth data: {e}")
        return {}, None

def transcribe_image(image, processor, model):
    """Run inference on a single image"""
    try:
        # Preprocess the image
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(device)
        
        # Generate text using the model
        outputs = model.generate(
            pixel_values,
            max_length=model.config.decoder.max_length,
            early_stopping=model.config.decoder.early_stopping,
            num_beams=model.config.decoder.num_beams,
            no_repeat_ngram_size=model.config.decoder.no_repeat_ngram_size,
        )
        
        # Decode the generated tokens
        transcription = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # Extract text between <s_text> and </s_text> tags if present
        if "<s_text>" in transcription and "</s_text>" in transcription:
            transcription = transcription.split("<s_text>")[1].split("</s_text>")[0]
        
        return transcription.strip()
    except Exception as e:
        print(f"Error during transcription: {e}")
        return f"ERROR: {str(e)}"

def calculate_word_error_rate(pred_text, true_text):
    """Calculate word error rate between predicted and true text"""
    if not true_text or true_text == "Ground truth text not found":
        return "N/A"
    
    pred_words = pred_text.split()
    true_words = true_text.split()
    
    # Calculate Levenshtein distance at word level
    if not true_words:
        return 1.0 if pred_words else 0.0
        
    m, n = len(pred_words), len(true_words)
    
    # Initialize distance matrix
    distance = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Base cases: empty true text
    for i in range(m + 1):
        distance[i][0] = i
    
    # Base cases: empty predicted text
    for j in range(n + 1):
        distance[0][j] = j
    
    # Fill the matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_words[i - 1] == true_words[j - 1]:
                distance[i][j] = distance[i - 1][j - 1]
            else:
                distance[i][j] = min(
                    distance[i - 1][j] + 1,  # deletion
                    distance[i][j - 1] + 1,  # insertion
                    distance[i - 1][j - 1] + 1  # substitution
                )
    
    # Calculate WER
    return distance[m][n] / n if n > 0 else 1.0

def calculate_character_error_rate(pred_text, true_text):
    """Calculate character error rate between predicted and true text"""
    if not true_text or true_text == "Ground truth text not found":
        return 1.0
    
    # Use torchmetrics if available
    try:
        cer_metric = CharErrorRate()
        return cer_metric(pred_text, true_text).item()
    except Exception as e:
        # Fallback to manual calculation if torchmetrics fails
        pred_chars = list(pred_text)
        true_chars = list(true_text)
        
        # Calculate Levenshtein distance at character level
        if not true_chars:
            return 1.0 if pred_chars else 0.0
            
        m, n = len(pred_chars), len(true_chars)
        
        # Initialize distance matrix
        distance = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Base cases
        for i in range(m + 1):
            distance[i][0] = i
        for j in range(n + 1):
            distance[0][j] = j
        
        # Fill the matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred_chars[i - 1] == true_chars[j - 1]:
                    distance[i][j] = distance[i - 1][j - 1]
                else:
                    distance[i][j] = min(
                        distance[i - 1][j] + 1,      # deletion
                        distance[i][j - 1] + 1,      # insertion
                        distance[i - 1][j - 1] + 1   # substitution
                    )
        
        # Calculate CER
        return distance[m][n] / n

def calculate_bleu_score(pred_text, true_text, ngrams=None):
    """Calculate BLEU score between predicted and true text
    
    Args:
        pred_text: Predicted text
        true_text: Ground truth text
        ngrams: Specific n-gram weights to use (e.g., (1,0,0,0) for unigram only)
               If None, use default weights
    """
    if not true_text or true_text == "Ground truth text not found":
        return 0.0
    
    # Try using the evaluate library
    try:
        import evaluate
        bleu = evaluate.load("bleu")
        
        if ngrams:
            # If specific n-gram weights are requested, use NLTK which supports custom weights
            raise ImportError("Using NLTK for custom n-gram weights")
            
        result = bleu.compute(predictions=[pred_text], references=[[true_text]])
        return result["bleu"]
    except:
        # Fallback to NLTK
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            import nltk
            # Download NLTK data if not already done
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
                
            # Tokenize the texts into words
            reference = [true_text.split()]
            hypothesis = pred_text.split()
            
            # Use smoothing function to avoid zero scores
            smoothie = SmoothingFunction().method1
            
            # Calculate BLEU score with specified weights or default to unigram only
            if ngrams:
                return sentence_bleu(reference, hypothesis, weights=ngrams, smoothing_function=smoothie)
            else:
                return sentence_bleu(reference, hypothesis, weights=(1, 0, 0, 0), smoothing_function=smoothie)
        except Exception as e:
            print(f"Error calculating BLEU score: {e}")
            return 0.0

def evaluate_random_samples(processor, model, ground_truth_dict, full_df, num_samples=100):
    """Evaluate the model on random samples from the dataset and save results to CSV"""
    print(f"\nEvaluating {num_samples} random samples...")
    
    # Try to install required packages if not already installed
    try:
        import nltk
        nltk.download('punkt', quiet=True)
    except:
        try:
            import subprocess
            subprocess.check_call(["pip", "install", "nltk", "torchmetrics", "evaluate", "sacrebleu"], 
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            import nltk
            nltk.download('punkt', quiet=True)
        except:
            print("Warning: Could not install required packages. Using fallback metrics.")
    
    # Select random samples from the dataframe
    if full_df is not None:
        # Prefer line images over full pages for better evaluation
        line_df = full_df[full_df['image_path'].str.contains('lines')]
        if len(line_df) >= num_samples:
            sample_df = line_df.sample(num_samples, random_state=42)
        else:
            # If not enough line images, take all lines and sample from the rest
            remaining = num_samples - len(line_df)
            other_df = full_df[~full_df['image_path'].str.contains('lines')].sample(remaining, random_state=42)
            sample_df = pd.concat([line_df, other_df])
        
        # Extract image paths and ground truths
        image_paths = sample_df['image_path'].tolist()
        random_images = [(path, os.path.basename(path)) for path in image_paths]
    else:
        # Fallback if dataframe not available: use dictionary keys
        all_images = list(ground_truth_dict.keys())
        # Filter out non-image entries (e.g., the path duplicates)
        all_images = [img for img in all_images if img.endswith('.jpg') or img.endswith('.png')]
        
        if len(all_images) > num_samples:
            sampled_images = random.sample(all_images, num_samples)
        else:
            sampled_images = all_images
        random_images = [(img, img) for img in sampled_images]
    
    results = []
    
    for i, (img_path, img_name) in enumerate(random_images):
        # Determine if it's a line image based on path/name
        is_line = "lines" in img_path or "_" in img_name
        
        # Find the image file
        image_path = find_image_path(img_name, is_line)
        if not image_path:
            print(f"Image {img_name} not found, skipping...")
            continue
        
        try:
            # Get ground truth
            ground_truth = ground_truth_dict.get(img_name, 
                                              ground_truth_dict.get(img_path, "Ground truth text not found"))
            
            # Load and prepare image
            image = Image.open(image_path).convert("RGB")
            
            # Process with finetuned model
            prediction = transcribe_image(image, processor, model)
            
            # Calculate metrics
            cer = calculate_character_error_rate(prediction, ground_truth)
            bleu = calculate_bleu_score(prediction, ground_truth)  # Default BLEU
            bleu_1 = calculate_bleu_score(prediction, ground_truth, ngrams=(1, 0, 0, 0))  # BLEU-1 (unigram only)
            bleu_2 = calculate_bleu_score(prediction, ground_truth, ngrams=(0.5, 0.5, 0, 0))  # BLEU-2 (unigram + bigram)
            
            # Store results
            results.append({
                "image_name": img_name,
                "ground_truth": ground_truth,
                "prediction_finetuned": prediction,
                "cer": cer,
                "bleu": bleu,
                "bleu_1": bleu_1,
                "bleu_2": bleu_2
            })
            
            # Print progress
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(random_images)} images")
        
        except Exception as e:
            print(f"Error processing image {img_name}: {e}")
    
    # Calculate average metrics
    if results:
        avg_cer = sum(result["cer"] for result in results) / len(results)
        avg_bleu = sum(result["bleu"] for result in results) / len(results)
        avg_bleu_1 = sum(result["bleu_1"] for result in results) / len(results)
        avg_bleu_2 = sum(result["bleu_2"] for result in results) / len(results)
        
        print(f"\nAverage CER: {avg_cer:.4f}")
        print(f"Average BLEU (default): {avg_bleu:.4f}")
        print(f"Average BLEU-1 (unigram): {avg_bleu_1:.4f}")
        print(f"Average BLEU-2 (unigram+bigram): {avg_bleu_2:.4f}")
    
    # Save results to CSV
    csv_output_path = os.path.join(os.getcwd(), "finetuned_model_evaluation.csv")
    with open(csv_output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['image_name', 'ground_truth', 'prediction_finetuned', 'cer', 'bleu', 'bleu_1', 'bleu_2']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"\nResults saved to: {csv_output_path}")
    
    return results

def main():
    # Load ground truth data
    ground_truth_dict, full_df = load_ground_truth_data()
    
    # Load the base model
    base_processor, base_model = load_model(BASE_MODEL)
    
    # Load the finetuned model if it exists
    if os.path.exists(FINETUNED_MODEL):
        print("\nLoading finetuned model...")
        finetuned_processor, finetuned_model = load_model(FINETUNED_MODEL)
    else:
        print(f"\nFinetuned model not found at: {FINETUNED_MODEL}")
        finetuned_processor, finetuned_model = None, None
    
    # Test with a set of images from the dataset
    test_images = [
        # Line images (from the CSV examples you provided)
        {"filename": "002_02_00.jpg", "is_line": True},
        {"filename": "002_02_01.jpg", "is_line": True},
        {"filename": "002_04_00.jpg", "is_line": True},
        {"filename": "002_04_01.jpg", "is_line": True},
        # Full page images (if you want to test those too)
        {"filename": "002_02.jpg", "is_line": False},
        {"filename": "002_04.jpg", "is_line": False},
    ]
    
    results = []
    
    for img_info in test_images:
        img_name = img_info["filename"]
        is_line = img_info["is_line"]
        
        image_path = find_image_path(img_name, is_line)
        if not image_path:
            continue
            
        try:
            # Load and prepare image
            image = Image.open(image_path).convert("RGB")
            print(f"\n{'='*50}")
            print(f"Processing image: {img_name}")
            
            # Get ground truth text from dictionary
            ground_truth = ground_truth_dict.get(img_name, "Ground truth text not found")
            if ground_truth == "Ground truth text not found" and is_line:
                # For line images, try looking for the relative path as in the CSV
                rel_path = f"lam_data/archive/LAM/lines/img/{img_name}"
                ground_truth = ground_truth_dict.get(rel_path, "Ground truth text not found")
            
            print(f"\nGround truth text: {ground_truth}")
            
            img_results = {
                "image": img_name,
                "ground_truth": ground_truth,
                "base_model": {},
                "finetuned_model": {}
            }
            
            # Test with base model
            if base_model is not None:
                print("\nRunning OCR with base model:")
                result = transcribe_image(image, base_processor, base_model)
                print(f"Base model output: {result}")
                
                # Calculate metrics
                wer = calculate_word_error_rate(result, ground_truth)
                cer = calculate_character_error_rate(result, ground_truth)
                bleu = calculate_bleu_score(result, ground_truth)
                bleu_1 = calculate_bleu_score(result, ground_truth, ngrams=(1, 0, 0, 0))
                bleu_2 = calculate_bleu_score(result, ground_truth, ngrams=(0.5, 0.5, 0, 0))
                
                # Display metrics
                wer_display = wer if isinstance(wer, str) else f"{wer:.2%}"
                print(f"Word Error Rate: {wer_display}")
                print(f"Character Error Rate: {cer:.2%}")
                print(f"BLEU: {bleu:.4f}, BLEU-1: {bleu_1:.4f}, BLEU-2: {bleu_2:.4f}")
                
                img_results["base_model"] = {
                    "text": result,
                    "wer": wer,
                    "cer": cer,
                    "bleu": bleu,
                    "bleu_1": bleu_1,
                    "bleu_2": bleu_2
                }
            
            # Test with finetuned model if available
            if finetuned_model is not None:
                print("\nRunning OCR with finetuned model:")
                result = transcribe_image(image, finetuned_processor, finetuned_model)
                print(f"Finetuned model output: {result}")
                
                # Calculate metrics
                wer = calculate_word_error_rate(result, ground_truth)
                cer = calculate_character_error_rate(result, ground_truth)
                bleu = calculate_bleu_score(result, ground_truth)
                bleu_1 = calculate_bleu_score(result, ground_truth, ngrams=(1, 0, 0, 0))
                bleu_2 = calculate_bleu_score(result, ground_truth, ngrams=(0.5, 0.5, 0, 0))
                
                # Display metrics
                wer_display = wer if isinstance(wer, str) else f"{wer:.2%}"
                print(f"Word Error Rate: {wer_display}")
                print(f"Character Error Rate: {cer:.2%}")
                print(f"BLEU: {bleu:.4f}, BLEU-1: {bleu_1:.4f}, BLEU-2: {bleu_2:.4f}")
                
                img_results["finetuned_model"] = {
                    "text": result,
                    "wer": wer,
                    "cer": cer,
                    "bleu": bleu,
                    "bleu_1": bleu_1,
                    "bleu_2": bleu_2
                }
            
            results.append(img_results)
            
        except Exception as e:
            print(f"Error processing image {img_name}: {e}")
    
    # Print overall comparison summary
    print("\n" + "="*80)
    print("OVERALL COMPARISON SUMMARY")
    print("="*80)
    
    base_wer_sum = 0
    base_wer_count = 0
    base_cer_sum = 0
    base_bleu_sum = 0
    base_bleu1_sum = 0
    base_bleu2_sum = 0
    
    finetuned_wer_sum = 0
    finetuned_wer_count = 0
    finetuned_cer_sum = 0
    finetuned_bleu_sum = 0
    finetuned_bleu1_sum = 0
    finetuned_bleu2_sum = 0
    
    for result in results:
        print(f"\nImage: {result['image']}")
        print(f"Ground Truth: {result['ground_truth']}")
        
        if result['base_model']:
            base_text = result['base_model']['text']
            base_wer = result['base_model']['wer']
            base_cer = result['base_model'].get('cer', 'N/A')
            base_bleu = result['base_model'].get('bleu', 'N/A')
            
            if not isinstance(base_wer, str):
                base_wer_sum += base_wer
                base_wer_count += 1
            
            if not isinstance(base_cer, str):
                base_cer_sum += base_cer
                
            if not isinstance(base_bleu, str):
                base_bleu_sum += base_bleu
                base_bleu1_sum += result['base_model'].get('bleu_1', 0)
                base_bleu2_sum += result['base_model'].get('bleu_2', 0)
                
            print(f"Base Model: {base_text}")
            print(f"  WER: {base_wer if isinstance(base_wer, str) else f'{base_wer:.2%}'}")
            print(f"  CER: {base_cer if isinstance(base_cer, str) else f'{base_cer:.2%}'}")
            print(f"  BLEU: {base_bleu if isinstance(base_bleu, str) else f'{base_bleu:.4f}'}")
        
        if result['finetuned_model']:
            finetuned_text = result['finetuned_model']['text']
            finetuned_wer = result['finetuned_model']['wer']
            finetuned_cer = result['finetuned_model'].get('cer', 'N/A')
            finetuned_bleu = result['finetuned_model'].get('bleu', 'N/A')
            
            if not isinstance(finetuned_wer, str):
                finetuned_wer_sum += finetuned_wer
                finetuned_wer_count += 1
                
            if not isinstance(finetuned_cer, str):
                finetuned_cer_sum += finetuned_cer
                
            if not isinstance(finetuned_bleu, str):
                finetuned_bleu_sum += finetuned_bleu
                finetuned_bleu1_sum += result['finetuned_model'].get('bleu_1', 0)
                finetuned_bleu2_sum += result['finetuned_model'].get('bleu_2', 0)
                
            print(f"Finetuned Model: {finetuned_text}")
            print(f"  WER: {finetuned_wer if isinstance(finetuned_wer, str) else f'{finetuned_wer:.2%}'}")
            print(f"  CER: {finetuned_cer if isinstance(finetuned_cer, str) else f'{finetuned_cer:.2%}'}")
            print(f"  BLEU: {finetuned_bleu if isinstance(finetuned_bleu, str) else f'{finetuned_bleu:.4f}'}")
    
    # Calculate averages
    count = len(results)
    avg_base_wer = base_wer_sum / base_wer_count if base_wer_count > 0 else "N/A"
    avg_base_cer = base_cer_sum / count if count > 0 else "N/A"
    avg_base_bleu = base_bleu_sum / count if count > 0 else "N/A"
    avg_base_bleu1 = base_bleu1_sum / count if count > 0 else "N/A"
    avg_base_bleu2 = base_bleu2_sum / count if count > 0 else "N/A"
    
    avg_finetuned_wer = finetuned_wer_sum / finetuned_wer_count if finetuned_wer_count > 0 else "N/A"
    avg_finetuned_cer = finetuned_cer_sum / count if count > 0 else "N/A"
    avg_finetuned_bleu = finetuned_bleu_sum / count if count > 0 else "N/A"
    avg_finetuned_bleu1 = finetuned_bleu1_sum / count if count > 0 else "N/A"
    avg_finetuned_bleu2 = finetuned_bleu2_sum / count if count > 0 else "N/A"
    
    print("\n" + "="*80)
    print("AVERAGE METRICS")
    print("="*80)
    print("Base Model:")
    print(f"  WER: {avg_base_wer if isinstance(avg_base_wer, str) else f'{avg_base_wer:.2%}'}")
    print(f"  CER: {avg_base_cer if isinstance(avg_base_cer, str) else f'{avg_base_cer:.2%}'}")
    print(f"  BLEU: {avg_base_bleu if isinstance(avg_base_bleu, str) else f'{avg_base_bleu:.4f}'}")
    print(f"  BLEU-1: {avg_base_bleu1 if isinstance(avg_base_bleu1, str) else f'{avg_base_bleu1:.4f}'}")
    print(f"  BLEU-2: {avg_base_bleu2 if isinstance(avg_base_bleu2, str) else f'{avg_base_bleu2:.4f}'}")
    
    print("\nFinetuned Model:")
    print(f"  WER: {avg_finetuned_wer if isinstance(avg_finetuned_wer, str) else f'{avg_finetuned_wer:.2%}'}")
    print(f"  CER: {avg_finetuned_cer if isinstance(avg_finetuned_cer, str) else f'{avg_finetuned_cer:.2%}'}")
    print(f"  BLEU: {avg_finetuned_bleu if isinstance(avg_finetuned_bleu, str) else f'{avg_finetuned_bleu:.4f}'}")
    print(f"  BLEU-1: {avg_finetuned_bleu1 if isinstance(avg_finetuned_bleu1, str) else f'{avg_finetuned_bleu1:.4f}'}")
    print(f"  BLEU-2: {avg_finetuned_bleu2 if isinstance(avg_finetuned_bleu2, str) else f'{avg_finetuned_bleu2:.4f}'}")
    
    # Calculate improvements
    if not isinstance(avg_base_wer, str) and not isinstance(avg_finetuned_wer, str):
        wer_improvement = (avg_base_wer - avg_finetuned_wer) / avg_base_wer * 100
        print(f"\nWER Improvement: {wer_improvement:.2f}%")
        
    if not isinstance(avg_base_cer, str) and not isinstance(avg_finetuned_cer, str):
        cer_improvement = (avg_base_cer - avg_finetuned_cer) / avg_base_cer * 100
        print(f"CER Improvement: {cer_improvement:.2f}%")
        
    if not isinstance(avg_base_bleu, str) and not isinstance(avg_finetuned_bleu, str):
        bleu_improvement = (avg_finetuned_bleu - avg_base_bleu) / max(0.001, avg_base_bleu) * 100
        print(f"BLEU Improvement: {bleu_improvement:.2f}%")
        
    print("="*80)
    
    # Evaluate 100 random samples with the finetuned model
    if finetuned_model is not None:
        print("\n" + "="*80)
        print("EVALUATING 100 RANDOM SAMPLES WITH FINETUNED MODEL")
        print("="*80)
        evaluate_random_samples(finetuned_processor, finetuned_model, ground_truth_dict, full_df, num_samples=100)

if __name__ == "__main__":
    main()