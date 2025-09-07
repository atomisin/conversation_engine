"""LoRA fine-tuning for negotiation phraser on a 13B model using QLoRA-style 4-bit loading.

Usage example:
  BASE_MODEL=meta-llama/Llama-2-13b       DATA=data/prompts_sample.jsonl       OUT=models/lora_13b       python services/conversation-engine/scripts/train_lora_13b.py
"""

import os, json
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

BASE_MODEL = os.getenv('BASE_MODEL', 'meta-llama/Llama-2-13b')
DATA_PATH = os.getenv('DATA', 'data/prompts_sample.jsonl')
OUT_DIR = os.getenv('OUT', 'models/lora_13b')

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)

def build_text(example):
    prompt = example['prompt'].strip()
    response = example['response'].strip()
    return f"<s>[INST] {prompt} [/INST] {response}</s>"

texts = [ {'text': build_text(x)} for x in load_jsonl(DATA_PATH) ]
from datasets import Dataset
dataset = Dataset.from_list(texts)
split = dataset.train_test_split(test_size=0.05, seed=42)
train_ds, eval_ds = split['train'], split['test']

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_4bit=True,
    device_map='auto',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'
)

model = prepare_model_for_kbit_training(model)

lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=['q_proj','k_proj','v_proj','o_proj'],
    lora_dropout=0.05,
    bias='none',
    task_type='CAUSAL_LM'
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

def tokenize(batch):
    return tokenizer(batch['text'], truncation=True, max_length=512)

train_tok = train_ds.map(tokenize, batched=True, remove_columns=['text'])
eval_tok  = eval_ds.map(tokenize, batched=True, remove_columns=['text'])

args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    per_device_eval_batch_size=1,
    evaluation_strategy='steps',
    eval_steps=200,
    save_steps=200,
    logging_steps=50,
    num_train_epochs=1,
    learning_rate=1e-4,
    fp16=torch.cuda.is_available(),
    report_to='none',
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_tok,
    eval_dataset=eval_tok,
    data_collator=data_collator
)

trainer.train()
trainer.save_model(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
print('LoRA training complete →', OUT_DIR)
