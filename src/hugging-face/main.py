from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

#if your model is saved in "model" folder
tokenizer = AutoTokenizer.from_pretrained("./model/")
model = AutoModelForSeq2SeqLM.from_pretrained("./model/")

# tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
# model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ru-en")

# Save the model and tokenizer to a directory
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


