import os
import torch
import pandas as pd
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

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
            print(f"Image found at: {path}")
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
                
                print(f"Loaded {len(ground_truth_dict)} ground truth entries")
                return ground_truth_dict
        
        print(f"Could not find {DATASET_CSV} file!")
        return {}
    except Exception as e:
        print(f"Error loading ground truth data: {e}")
        return {}

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

def main():
    # Load ground truth data
    ground_truth_dict = load_ground_truth_data()
    
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
                wer = calculate_word_error_rate(result, ground_truth)
                wer_display = wer if isinstance(wer, str) else f"{wer:.2%}"
                print(f"Word Error Rate: {wer_display}")
                
                img_results["base_model"] = {
                    "text": result,
                    "wer": wer
                }
            
            # Test with finetuned model if available
            if finetuned_model is not None:
                print("\nRunning OCR with finetuned model:")
                result = transcribe_image(image, finetuned_processor, finetuned_model)
                print(f"Finetuned model output: {result}")
                wer = calculate_word_error_rate(result, ground_truth)
                wer_display = wer if isinstance(wer, str) else f"{wer:.2%}"
                print(f"Word Error Rate: {wer_display}")
                
                img_results["finetuned_model"] = {
                    "text": result,
                    "wer": wer
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
    finetuned_wer_sum = 0
    finetuned_wer_count = 0
    
    for result in results:
        print(f"\nImage: {result['image']}")
        print(f"Ground Truth: {result['ground_truth']}")
        
        if result['base_model']:
            base_text = result['base_model']['text']
            base_wer = result['base_model']['wer']
            if not isinstance(base_wer, str):
                base_wer_sum += base_wer
                base_wer_count += 1
            print(f"Base Model: {base_text} (WER: {base_wer if isinstance(base_wer, str) else f'{base_wer:.2%}'})")
        
        if result['finetuned_model']:
            finetuned_text = result['finetuned_model']['text']
            finetuned_wer = result['finetuned_model']['wer']
            if not isinstance(finetuned_wer, str):
                finetuned_wer_sum += finetuned_wer
                finetuned_wer_count += 1
            print(f"Finetuned Model: {finetuned_text} (WER: {finetuned_wer if isinstance(finetuned_wer, str) else f'{finetuned_wer:.2%}'})")
    
    # Calculate average WER
    avg_base_wer = base_wer_sum / base_wer_count if base_wer_count > 0 else "N/A"
    avg_finetuned_wer = finetuned_wer_sum / finetuned_wer_count if finetuned_wer_count > 0 else "N/A"
    
    print("\n" + "="*80)
    print(f"Average Base Model WER: {avg_base_wer if isinstance(avg_base_wer, str) else f'{avg_base_wer:.2%}'}")
    print(f"Average Finetuned Model WER: {avg_finetuned_wer if isinstance(avg_finetuned_wer, str) else f'{avg_finetuned_wer:.2%}'}")
    if not isinstance(avg_base_wer, str) and not isinstance(avg_finetuned_wer, str):
        improvement = (avg_base_wer - avg_finetuned_wer) / avg_base_wer * 100
        print(f"Improvement: {improvement:.2f}%")
    print("="*80)

if __name__ == "__main__":
    main()