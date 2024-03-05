from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AdamW, get_linear_schedule_with_warmup
import torch
import numpy as np
import sentencepiece as spm


#if your model is saved in "model" folder
tokenizer = AutoTokenizer.from_pretrained("./model/")
model = AutoModelForSeq2SeqLM.from_pretrained("./model/")

# Save the model and tokenizer to a directory

# tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
# model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
# model.save_pretrained('./model')
# tokenizer.save_pretrained('./model/')



def translate(text, model, tokenizer):
    # Tokenize the input text and return PyTorch tensors
    inputs = tokenizer.encode(text, return_tensors="pt")
    
    # Generate translation using model
    outputs = model.generate(inputs, max_length=40, num_beams=4, early_stopping=True)
    
    # Decode the generated tokens to string
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return translated_text


russian_text = "Привет, как дела?"
translated_text = translate(russian_text, model, tokenizer)
print(f"Translated Text: {translated_text}")


#freeze the decoder (we save encoder since Russian and Cherokee are similar)
for param in model.model.encoder.parameters():
    param.requires_grad = False


class CherokeeEnglishDataset(Dataset):
    def __init__(self, source_file, target_file, source_tokenizer, target_tokenizer, max_length=512):
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.max_length = max_length
        self.inputs = []
        self.targets = []

        with open(source_file, encoding="utf-8") as src_file, open(target_file, encoding="utf-8") as tgt_file:
            for src_line, tgt_line in zip(src_file, tgt_file):
                src_encoded = source_tokenizer.EncodeAsIds(src_line.strip())
                tgt_encoded = target_tokenizer.EncodeAsIds(tgt_line.strip())
                # Ensure the encoded lists are not longer than max_length
                src_encoded = src_encoded[:max_length - 1]
                tgt_encoded = tgt_encoded[:max_length - 1]
                self.inputs.append(torch.tensor(src_encoded))
                self.targets.append(torch.tensor(tgt_encoded))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            "input_ids": self.inputs[idx],
            "labels": self.targets[idx]
        }


# Initialize SentencePiece tokenizers
source_sp_model = spm.SentencePieceProcessor(model_file='source.model')
target_sp_model = spm.SentencePieceProcessor(model_file='target.model')

# Adjust file paths as needed
train_dataset = CherokeeEnglishDataset('chr-en.filtered.chr', 'chr-en.filtered.en', source_sp_model, target_sp_model)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)  # Adjust batch size as needed

# Initialize the model
model = AutoModelForSeq2SeqLM.from_pretrained("./model/")

# Transfer the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Setup optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)  # Adjust learning rate as needed

# Setup training
num_epochs = 3  # Adjust epochs as needed
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# Training loop
model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

# Save the fine-tuned model
model.save_pretrained('./fine_tuned_model')