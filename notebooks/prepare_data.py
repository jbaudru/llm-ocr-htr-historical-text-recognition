import json
import os
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
import pandas as pd

# Load the train and test data
with open('lam_data/archive/LAM/lines/split/basic/train.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

with open('lam_data/archive/LAM/lines/split/basic/test.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

# Base path for the images
base_img_path = 'lam_data/archive/LAM/lines/img/'

# Create a list to store the formatted data
formatted_data = []

# Process training data
for item in train_data:
    image_path = os.path.join(base_img_path, item['img'])
    
    # Check if image exists
    if os.path.exists(image_path):
        formatted_data.append({
            'image_path': image_path,
            'text': item['text'],
            'split': 'train',
            'decade_id': item['decade_id']
        })

# Process test data
for item in test_data:
    image_path = os.path.join(base_img_path, item['img'])
    
    # Check if image exists
    if os.path.exists(image_path):
        formatted_data.append({
            'image_path': image_path,
            'text': item['text'],
            'split': 'test',
            'decade_id': item['decade_id']
        })

# Save as DataFrame for easier processing
df = pd.DataFrame(formatted_data)
df.to_csv('handwritten_dataset.csv', index=False)

print(f"Prepared dataset with {len(df)} entries")
print(f"Train samples: {len(df[df['split'] == 'train'])}")
print(f"Test samples: {len(df[df['split'] == 'test'])}")