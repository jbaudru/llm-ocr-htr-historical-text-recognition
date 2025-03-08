import argparse
import time
import json
import os
import pandas as pd
from PIL import Image as PILImage
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AdamW
from tqdm import tqdm

parser = argparse.ArgumentParser(description="LAM Verification Script")
parser.add_argument("--end_epoch", type=int, default=20, help="Number of epochs to run")
args = parser.parse_args()


img_path = '/data/brussel/vo/000/bvo00042/img'

with open('transcriptions.json', 'r') as f:
    tr = json.load(f)

transcriptions = pd.DataFrame(tr)

train = transcriptions[transcriptions['nameset'] == 'train'].reset_index(drop=True)
test = transcriptions[transcriptions['nameset'] == 'test'].reset_index(drop=True)
print(f'\nTrain set length: {len(train)} \nTest set length: {len(test)}')
print(f'% train: {(len(train)/(len(test)+len(train)))*100}')

class IAMDatasetNew(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df['img'][idx]
        text = self.df['text'][idx]
        image = PILImage.open(self.root_dir +'/'+ file_name).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        labels = self.processor.tokenizer(text, padding="max_length", max_length=self.max_target_length).input_ids
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten", use_fast=False)
train_dataset = IAMDatasetNew(root_dir=img_path, df=train, processor=processor)
eval_dataset = IAMDatasetNew(root_dir=img_path, df=test, processor=processor)

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
eval_dataloader = DataLoader(eval_dataset, batch_size=4, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1").to(device)

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = 200
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4

torch.cuda.empty_cache()

output_folder = "/data/brussel/vo/000/bvo00042/output_serene_3"
os.makedirs(output_folder, exist_ok=True)
checkpoint_path = os.path.join(output_folder, "checkpoint.pth")

optimizer = AdamW(model.parameters(), lr=5e-5)

start_epoch = 0
end_epoch = args.end_epoch

for epoch in range(start_epoch, end_epoch):
    print(f"Starting epoch {epoch}")
    model.train()
    train_loss = 0.0
    for batch in tqdm(train_dataloader):
        for k, v in batch.items():
            batch[k] = v.to(device)

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()

    print(f"Loss after epoch {epoch}:", train_loss / len(train_dataloader))
    
    model.eval()
    epoch_output_folder = f'{output_folder}/epoch{epoch}'
    os.makedirs(epoch_output_folder, exist_ok=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_dataloader)):
            print(f'Generating text for epoch {epoch}, batch {batch_idx}')
            outputs = model.generate(batch["pixel_values"].to(device))
            decoded_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

            find_filename = test['img'][batch_idx] if batch_idx < len(test) else f'unknown_{batch_idx}'
            output_path = os.path.join(epoch_output_folder, f'{find_filename}_epoch_{epoch}_batch_{batch_idx}.txt')
            with open(output_path, "w") as f:
                f.write(decoded_text)

            del outputs, decoded_text

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

model.save_pretrained(output_folder)
print("✅ Finished Training and Saving the model: time taken =", time.time() - start_time)

# checkpoint_path = os.path.join(output_folder, "checkpoint.pth")
# torch.save({
#     "epoch": epoch,
#     "model_state_dict": model.state_dict(),
#     "optimizer_state_dict": optimizer.state_dict(),
#     "loss": train_loss
# }, checkpoint_path)