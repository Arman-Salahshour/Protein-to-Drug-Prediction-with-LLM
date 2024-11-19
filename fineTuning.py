import os
import torch
import data_builder
from typing import Optional
from dataclasses import dataclass
from trl import SFTTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    EarlyStoppingCallback,
)
from peft import LoraConfig, PeftModel, get_peft_model

# Paths and model information
model_name = 'mistralai/Mistral-7B-v0.3'
version = 1  # Version of the fine-tuning

# Paths to dataset files
training_data_path = "./dataset/train.da"
validation_data_path = "./dataset/validation.da"
test_data_path = "./dataset/test.da"

# Tokenizer setup
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.padding_side = 'right'

# Load datasets; if not available, build and save them
if os.path.exists(training_data_path):
    dataset_train = torch.load(training_data_path)
    dataset_validation = torch.load(validation_data_path)
else:
# Data building process
    dataset = data_builder.BuildTCREpitope(tokenizer)
    torch.save(dataset.hf_dataset_train, training_data_path)
    torch.save(dataset.hf_dataset_validation, validation_data_path)
    torch.save(dataset.hf_dataset_test, test_data_path)
    dataset_train = dataset.hf_dataset_train
    dataset_validation = dataset.hf_dataset_validation
    

# Training configuration
epochs = 20 
batch_size = 2
eval_batch_size = 2
accumulation_steps = 32

# PEFT (LoRA) configuration parameters
lora_r = 128
lora_alpha = 16
lora_dropout = 0.1

# Quantization settings
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False
max_seq_length = 2048
device_map = 'auto'

# Create TrainingArguments object
training_arguments = TrainingArguments(
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=eval_batch_size,
    gradient_accumulation_steps=accumulation_steps,
    warmup_steps=int(0.3 * (epochs * len(dataset_train) // (batch_size * accumulation_steps))),
    num_train_epochs=epochs,
    learning_rate=2e-4,
    fp16= True,
    bf16= False,
    logging_steps=1,
    eval_strategy="epoch",
    optim="paged_adamw_32bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir=f"outputs_MistralV_justTCREpitope{version}",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# Quantization configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant
)

# Load the model with quantization configuration and disable cache use
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map=device_map)
model.config.use_cache = False
model.config.pretraining_tp = 1

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# PEFT (LoRA) configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias='none',
    task_type='CAUSAL_LM',
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

# Wrap the model with PEFT model
model = get_peft_model(model, peft_config)

# Initialize the trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset_train,
    eval_dataset=dataset_validation,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=training_arguments,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)

# Start training
trainer_stats = trainer.train()

model.save_pretrained(f"lora_model_{model_name}V{version}") # Local saving
tokenizer.save_pretrained(f"lora_model_{model_name}V{version}")