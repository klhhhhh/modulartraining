from transformers import BertForMaskedLM
from transformers import BertConfig
from transformers import BertTokenizer
import datasets
import json
import sys
def tokenize_function(examples):
    return tokenizer(examples["text"])

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',use_fast=True) 
model = BertForMaskedLM.from_pretrained('/home/wanzhipeng/deepincubation/BertBase/checkpoint-18500')
# model_checkpoint = "distilroberta-base"
# tokenizer = BertTokenizer.from_pretrained(model_checkpoint, use_fast=True)
# tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
# model = BertForMaskedLM.from_pretrained(model_checkpoint)
from datasets import load_dataset
# dataset = load_dataset("cc_news", split="train")
datasets = load_dataset('wikitext', 'wikitext-2-raw-v1')

tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

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
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)



from transformers import pipeline
unmasker = pipeline('fill-mask', model=model.to("cpu"),tokenizer=tokenizer)
unmasker("Hello I'm a [MASK] model.")
print(unmasker)






