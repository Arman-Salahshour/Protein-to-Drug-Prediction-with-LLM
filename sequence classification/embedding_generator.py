import os
import torch
import zipfile
import data_builder
from tqdm import tqdm
from datasets import load_metric
from torch.utils.data import DataLoader
from peft import PeftModel, PeftModelForSequenceClassification
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, Trainer, TrainingArguments

def make_zip(path):
    zip_path = '.'.join(path.split('.')[:-1]) + '.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(path, arcname=path)
        
    os.remove(path)
                
#Load tokenizer and model
check_point = '/mnt/newssd2/amir/Arman/LLM/outputs_MistralV_ProteinTCREpitope1/checkpoint-5160'

# Tokenizer setup
tokenizer = AutoTokenizer.from_pretrained(check_point, trust_remote_code=True)
tokenizer.padding_side = 'right'
model = AutoModel.from_pretrained(check_point,  device_map='auto')

# Ensure the tokenizer has the correct special tokens and resize embeddings
model.resize_token_embeddings(len(tokenizer))


# Paths to dataset files
embedding_path = '/mnt/newssd2/amir/Arman/LLM/sequence classification/embedding'
dataset_path = '/mnt/newssd2/amir/Arman/LLM/sequence classification/dataset'
training_data_path = os.path.join(dataset_path, "train.da")
validation_data_path = os.path.join(dataset_path, "validation.da")
test_data_path = os.path.join(dataset_path, "test.da")

# Load datasets; if not available, build and save them
if os.path.exists(training_data_path):
    dataset_train = torch.load(training_data_path)
    dataset_validation = torch.load(validation_data_path)
    dataset_test= torch.load(test_data_path)
else:
# Data building process
    dataset = data_builder.BuildTCREpitope(tokenizer)
    torch.save(dataset.hf_dataset_train, training_data_path)
    torch.save(dataset.hf_dataset_validation, validation_data_path)
    torch.save(dataset.hf_dataset_test, test_data_path)
    dataset_train = dataset.hf_dataset_train
    dataset_validation = dataset.hf_dataset_validation
    dataset_test = dataset.hf_dataset_test


def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', max_length=523, truncation=True)

tokenized_train = dataset_train.map(tokenize_function, batched=True)
tokenized_validation = dataset_validation.map(tokenize_function, batched=True)
tokenized_test = dataset_test.map(tokenize_function, batched=True)

tokenized_train.set_format('torch', columns=["input_ids", "attention_mask", "label"])
tokenized_validation.set_format('torch', columns=["input_ids", "attention_mask", "label"])
tokenized_test.set_format('torch', columns=["input_ids", "attention_mask", "label"])


train_dataloader = DataLoader(tokenized_train, batch_size=32, shuffle=True)
validation_dataloader = DataLoader(tokenized_validation, batch_size=32, shuffle=True)
test_dataloader = DataLoader(tokenized_test, batch_size=32, shuffle=True)

data_loaders_list = {
    'training':train_dataloader, 
    'validation':validation_dataloader, 
    'test':test_dataloader
}

for item in data_loaders_list.items():
    phase = item[0]
    dataloader = item[1]

    # Initialize the progress bar with the total number of batches
    total_steps = len(dataloader)
    progress_bar = tqdm(total=total_steps, desc=f"Generating {phase} embeddings")

    embeddings = None
    labels = None
    chunk_counter = 1
    for i, batch in enumerate(dataloader):
        
        if embeddings is None:
            embeddings, _ = torch.max(model(batch['input_ids'], batch['attention_mask']).last_hidden_state, dim=1)
            labels = batch['label']
        else:
            embd, _ = torch.max(model(batch['input_ids'], batch['attention_mask']).last_hidden_state, dim=1)
            embeddings = torch.cat((embeddings, embd), dim=0)
            labels = torch.cat((labels, batch['label']), dim=0)

        progress_bar.update(1)  # Update the progress bar by one step
        
        if (i + 1) % 80 == 0:
            torch.save(embeddings, os.path.join(embedding_path, f'{phase}{chunk_counter}.embd'))
            # make_zip(os.path.join(embedding_path, f'{phase}.embd'))
            
            torch.save(labels, os.path.join(embedding_path, f'{phase}{chunk_counter}_labels.embd'))
            # make_zip(os.path.join(embedding_path, f'{phase}_labels.embd'))
            
            embeddings = None
            labels = None
            chunk_counter += 1
    

    torch.save(embeddings, os.path.join(embedding_path, f'{phase}{chunk_counter}.embd'))
    # make_zip(os.path.join(embedding_path, f'{phase}.embd'))
    
    torch.save(labels, os.path.join(embedding_path, f'{phase}{chunk_counter}_labels.embd'))
    # make_zip(os.path.join(embedding_path, f'{phase}_labels.embd'))
    
