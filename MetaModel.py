#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import BertForMaskedLM
from transformers import BertConfig
from transformers import BertTokenizer
import datasets
import json
import sys
def tokenize_function(examples):
    return tokenizer(examples["text"])
config = BertConfig(num_hidden_layers=4)
model = BertForMaskedLM(config)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',use_fast=True) 
model



# In[2]:


from datasets import load_dataset
datasets = load_dataset('wikitext', 'wikitext-2-raw-v1')
tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])



# In[3]:


block_size = 128
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result
lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=4,
)


# In[9]:


from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
training_args = TrainingArguments(
    output_dir="MetaModel_bert_wiki",          # output directory to where save model checkpoint
    evaluation_strategy="steps",    # evaluate each `logging_steps` steps
    logging_strategy="steps",
    overwrite_output_dir=True,      
    num_train_epochs=500,            # number of training epochs, feel free to tweak
    # per_device_train_batch_size=1000, # the training batch size, put it as high as your GPU memory fits
    # gradient_accumulation_steps=8,  # accumulating the gradients before updating the weights
    # per_device_eval_batch_size=1000,  # evaluation batch size
    logging_steps=500,             # evaluate, log and save model checkpoints every 1000 step
    save_steps=500,
    load_best_model_at_end=True,  # whether to load the best model (in terms of loss) at the end of training
    save_total_limit=3,           # whether you don't have much space so you let only 3 model weights saved in the disk
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_steps=10000,
    per_device_train_batch_size=32,
    per_gpu_eval_batch_size=32,
)

trainer = Trainer(
    model=model.to("cuda"),
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
    data_collator=data_collator,
)




# In[10]:


trainer.train(resume_from_checkpoint=True)

