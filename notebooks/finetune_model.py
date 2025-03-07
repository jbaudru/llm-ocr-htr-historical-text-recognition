import os
import random
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
import evaluate
import jiwer  # For WER/CER calculations

from transformers import (
    DonutProcessor, 
    VisionEncoderDecoderConfig,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments,
    default_data_collator
)

# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed()

# Configuration
DATA_PATH = "handwritten_dataset.csv"
BASE_PATH = ""
OUTPUT_DIR = os.path.join(BASE_PATH, "models/finetuned_donut_vlm")
CACHE_DIR = os.path.join(BASE_PATH, "models/cache")
MODEL_NAME = "naver-clova-ix/donut-base"  # A small efficient VLM for document understanding
SAMPLE_FRAC = 1.0  # Fraction of data to use for training (1.0 = full dataset) 
BATCH_SIZE = 1      # Use batch size of 1 to avoid shape issues
LEARNING_RATE = 3e-5
NUM_EPOCHS = 4
MAX_LENGTH = 128   # Maximum sequence length for tokenization

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Custom CER function (since evaluate.load("cer") is failing)
def calculate_cer(predictions, references):
    """Calculate character error rate"""
    total_cer = 0
    for pred, ref in zip(predictions, references):
        # Fallback to library implementation if available
        try:
            if jiwer is not None:
                # Use jiwer for CER calculation
                cer = jiwer.cer(ref, pred)
                total_cer += cer
                continue
        except:
            pass

        # Manual CER calculation if jiwer fails
        distance = levenshtein_distance(ref, pred)
        length = max(len(ref), 1)  # Avoid division by zero
        total_cer += distance / length
    
    return total_cer / max(len(predictions), 1)

# Custom WER function (since evaluate.load("wer") is failing)
def calculate_wer(predictions, references):
    """Calculate word error rate"""
    total_wer = 0
    for pred, ref in zip(predictions, references):
        # Fallback to library implementation if available
        try:
            if jiwer is not None:
                # Use jiwer for WER calculation
                wer = jiwer.wer(ref, pred)
                total_wer += wer
                continue
        except:
            pass
            
        # Manual WER calculation if jiwer fails
        pred_words = pred.split()
        ref_words = ref.split()
        distance = levenshtein_distance(ref_words, pred_words)
        length = max(len(ref_words), 1)  # Avoid division by zero
        total_wer += distance / length
    
    return total_wer / max(len(predictions), 1)

# Levenshtein distance (edit distance) for CER/WER calculation
def levenshtein_distance(s1, s2):
    """Calculate the Levenshtein distance between two strings or lists"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

# Data loading function
def load_data(data_path, base_path, sample_fraction=0.5, max_examples=None):
    """Load dataset and sample a percentage of the data with optional maximum"""
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Make image paths absolute
    df['image_path'] = df['image_path'].apply(lambda x: os.path.join(base_path, x))
    
    # Verify images exist
    print(f"Verifying images exist...")
    df['image_exists'] = df['image_path'].apply(os.path.isfile)
    missing_count = sum(~df['image_exists'])
    
    if missing_count > 0:
        print(f"Warning: {missing_count} images not found!")
        print(f"First few missing images: {df[~df['image_exists']]['image_path'].values[:5]}")
        df = df[df['image_exists']]
    
    df = df.drop('image_exists', axis=1)
    
    # Increased sample sizes for more robust training
    train_df = df[df['split'] == 'train'].sample(min(400, len(df[df['split'] == 'train'])), random_state=42)
    val_df = df[df['split'] == 'val'].sample(min(40, len(df[df['split'] == 'val'])), random_state=42) if 'val' in df['split'].values else None
    test_df = df[df['split'] == 'test'].sample(min(40, len(df[df['split'] == 'test'])), random_state=42) if 'test' in df['split'].values else None
    
    # If no explicit validation set, create one from train
    if val_df is None or len(val_df) == 0:
        if len(train_df) > 25:
            train_df, val_df = train_test_split(train_df, test_size=min(40, len(train_df)//5), random_state=42)
        else:
            # If train_df is too small, just use the same data for validation
            val_df = train_df.copy()
    
    # If no explicit test set, use validation as test
    if test_df is None or len(test_df) == 0:
        test_df = val_df.copy()
        
    print(f"Using {len(train_df)} training examples")
    print(f"Using {len(val_df)} validation examples")
    print(f"Using {len(test_df)} test examples")
            
    return train_df, val_df, test_df

# Convert dataframes to HF Datasets
def create_dataset_dict(train_df, val_df, test_df):
    """Convert dataframes to HuggingFace Dataset format"""
    
    # Drop unnecessary columns
    keep_columns = ['image_path', 'text']
    train_df = train_df[keep_columns].copy()
    val_df = val_df[keep_columns].copy()
    test_df = test_df[keep_columns].copy()
    
    # Create HF datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    return DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })

# Process dataset for Donut model
def process_dataset(dataset_dict, processor):
    """Process the dataset to prepare it for training with Donut"""
    
    # Define preprocessing function
    def preprocess_data(example):
        # Format text for Donut
        text = f"<s_text>{example['text']}</s_text>"
        
        # Process image
        try:
            image = Image.open(example["image_path"]).convert("RGB")
        except Exception as e:
            print(f"Error opening image {example['image_path']}: {e}")
            # Use a blank image as fallback
            image = Image.new('RGB', (100, 32), color='white')
        
        # Process inputs - use legacy=False to avoid warnings
        encoding = processor(
            images=image, 
            text=text, 
            padding="max_length",
            truncation=True, 
            max_length=MAX_LENGTH,
            return_tensors="pt",
            legacy=False
        )
        
        # Remove batch dimension
        for k, v in encoding.items():
            if isinstance(v, torch.Tensor):
                encoding[k] = v.squeeze(0)
        
        # Replace padding token id with -100 for labels (if labels exist)
        if 'labels' in encoding:
            labels = encoding['labels'].clone()
            labels[labels == processor.tokenizer.pad_token_id] = -100
            encoding['labels'] = labels
        
        return encoding

    # Apply preprocessing to each split
    processed_datasets = {}
    
    for split, dataset in dataset_dict.items():
        print(f"Processing {split} dataset...")
        processed = dataset.map(
            preprocess_data,
            remove_columns=["image_path", "text"],
            desc=f"Processing {split} dataset",
        )
        
        # Check what columns are actually in the processed dataset
        available_columns = processed.column_names
        print(f"Available columns in processed {split} dataset: {available_columns}")
        
        # Set format only for columns that exist
        tensor_columns = []
        for col in ['pixel_values', 'labels', 'decoder_input_ids', 'attention_mask']:
            if col in available_columns:
                tensor_columns.append(col)
                
        processed.set_format(type="torch", columns=tensor_columns)
        processed_datasets[split] = processed
    
    return DatasetDict(processed_datasets)

# Metrics for evaluation
def compute_metrics(eval_pred):
    """Compute metrics for model evaluation"""
    predictions, labels = eval_pred
    
    # Replace negative values in predictions with pad token ID
    predictions = np.where(predictions >= 0, predictions, processor.tokenizer.pad_token_id)
    
    # Replace -100 in labels with pad token ID
    labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
    
    # Decode predictions and references
    try:
        decoded_preds = processor.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)
    except Exception as e:
        print(f"Error during batch_decode: {e}")
        print(f"Min prediction value: {np.min(predictions)}, Max: {np.max(predictions)}")
        print(f"Min label value: {np.min(labels)}, Max: {np.max(labels)}")
        # Return worst case metrics if decoding fails
        return {"cer": 1.0, "wer": 1.0, "exact_match": 0.0}
    
    # Extract text from the formatted strings
    def extract_text(text):
        if "<s_text>" in text and "</s_text>" in text:
            return text.split("<s_text>")[1].split("</s_text>")[0].strip()
        return text.strip()
    
    # Clean up predictions and references
    decoded_preds = [extract_text(pred) for pred in decoded_preds]
    decoded_labels = [extract_text(label) for label in decoded_labels]
    
    # Use custom CER and WER functions instead of evaluate library
    print("Computing CER and WER with custom functions")
    try:
        cer = calculate_cer(decoded_preds, decoded_labels)
        wer = calculate_wer(decoded_preds, decoded_labels)
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        cer = 1.0  # Fallback to worst-case value
        wer = 1.0
    
    # Compute exact match
    exact_matches = sum(pred == label for pred, label in zip(decoded_preds, decoded_labels))
    exact_match_rate = exact_matches / len(decoded_preds) if len(decoded_preds) > 0 else 0
    
    return {
        "cer": cer,
        "wer": wer,
        "exact_match": exact_match_rate
    }

class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    """Custom trainer to avoid the num_items_in_batch issue"""
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Remove num_items_in_batch from inputs to avoid the error
        if isinstance(inputs, dict) and "num_items_in_batch" in inputs:
            del inputs["num_items_in_batch"]
            
        # Standard loss computation
        outputs = model(**inputs)
        loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss

def main():
    global processor  # Make processor accessible for compute_metrics
    
    # Load and sample data
    print("Loading and sampling data...")
    train_df, val_df, test_df = load_data(DATA_PATH, BASE_PATH, SAMPLE_FRAC)
    
    # Create dataset dictionary
    print("Creating dataset dictionary...")
    dataset_dict = create_dataset_dict(train_df, val_df, test_df)
    
    # Initialize processor and model
    print(f"Loading Donut model {MODEL_NAME}...")
    processor = DonutProcessor.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    
    # Configure processor and model
    processor.image_processor.size = {"height": 480, "width": 960}
    
    # Add special tokens if needed and ensure they're already in tokenizer
    special_tokens = ["<s_text>", "</s_text>"]
    if special_tokens[0] not in processor.tokenizer.get_vocab():
        processor.tokenizer.add_tokens(special_tokens)
        model.decoder.resize_token_embeddings(len(processor.tokenizer))
    
    # Set up model configuration
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(["<s_text>"])[0]
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = len(processor.tokenizer)
    
    # Set special generation parameters
    model.config.decoder.min_length = 1
    model.config.decoder.max_length = MAX_LENGTH
    model.config.decoder.no_repeat_ngram_size = 3
    model.config.decoder.early_stopping = True
    model.config.decoder.num_beams = 4
    
    # Process datasets
    print("Processing datasets...")
    processed_datasets = process_dataset(dataset_dict, processor)
    
    # Inspect dataset format
    print("\nSample processed dataset item:")
    sample_item = processed_datasets["train"][0]
    for key, value in sample_item.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: tensor of shape {value.shape}")
        else:
            print(f"{key}: {type(value)}")
    
    # Configure training arguments for tiny dataset
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="steps",
        eval_steps=10,
        save_steps=10,
        logging_steps=1,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=NUM_EPOCHS,
        predict_with_generate=True,
        generation_max_length=MAX_LENGTH,
        fp16=False,
        gradient_accumulation_steps=1,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        report_to="none",
        remove_unused_columns=False,
    )
    
    # Initialize trainer
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_datasets["train"],
        eval_dataset=processed_datasets["validation"],
        compute_metrics=compute_metrics,
        data_collator=default_data_collator,
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the model
    print(f"Saving model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = trainer.evaluate(processed_datasets["test"], metric_key_prefix="test")
    print(f"Test results: {test_results}")
    
    # Write test results to file
    with open(os.path.join(OUTPUT_DIR, "test_results.txt"), "w") as f:
        f.write(str(test_results))
    
    # Test inference on examples
    print("\nTesting inference on examples:")
    
    def transcribe_image(image_path):
        """Run inference on a single image"""
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt", legacy=False)
        
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            model.to("cuda")
            
        # Generate text using the model
        outputs = model.generate(
            inputs["pixel_values"],
            max_length=model.config.decoder.max_length,
            early_stopping=model.config.decoder.early_stopping,
            num_beams=model.config.decoder.num_beams,
            no_repeat_ngram_size=model.config.decoder.no_repeat_ngram_size,
        )
        
        # Decode the generated tokens
        transcription = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # Extract text between <s_text> and </s_text> tags
        if "<s_text>" in transcription and "</s_text>" in transcription:
            transcription = transcription.split("<s_text>")[1].split("</s_text>")[0]
        
        return transcription.strip()
    
    # Try inference on examples
    for i in range(min(5, len(test_df))):
        image_path = test_df.iloc[i]['image_path']
        expected_text = test_df.iloc[i]['text']
        
        predicted_text = transcribe_image(image_path)
        
        print(f"\nExample {i+1}:")
        print(f"Image path: {os.path.basename(image_path)}")
        print(f"Predicted: {predicted_text}")
        print(f"Expected:  {expected_text}")
        
        # Calculate character error rate for this example
        example_cer = calculate_cer([predicted_text], [expected_text])
        print(f"CER: {example_cer:.4f}")

if __name__ == "__main__":
    main()