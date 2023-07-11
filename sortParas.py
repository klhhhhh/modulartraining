# %%
from transformers import BertForMaskedLM
from transformers import BertConfig
from transformers import BertTokenizer
import datasets
import json
import sys
import nni

# %%
def tokenize_function(examples):
    return tokenizer(examples["text"])
config = BertConfig()
model = BertForMaskedLM(config)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',use_fast=True) 
# model = BertForMaskedLM.from_pretrained('/home/wanzhipeng/deepincubation/BertBase/checkpoint-21500')


# %%
from datasets import load_dataset
datasets = load_dataset('wikitext', 'wikitext-2-raw-v1')

tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])



# %%
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
    num_proc=8,
).shuffle(seed=42)


# %%
from transformers import Trainer, TrainingArguments,EarlyStoppingCallback,TrainerCallback
from transformers import DataCollatorForLanguageModeling
# 自动参数调优
params = {
    'weight_decay': 0.01,
    'lr': 1e-5,
}
optimized_params = nni.get_next_parameter()
params.update(optimized_params)
loss = None
class CallbackForNNI(TrainerCallback):

    def on_evaluate(self, args, state, control, metrics=None,**kwargs):
        global loss
        loss = metrics["eval_loss"]
        nni.report_intermediate_result(loss)
        
        
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
training_args = TrainingArguments(
    output_dir="BertBase",          # output directory to where save model checkpoint
    evaluation_strategy="epoch",    # evaluate each `logging_steps` steps
    logging_strategy="epoch",
    save_strategy="epoch",
    overwrite_output_dir=True,      
    num_train_epochs=5,            # number of training epochs, feel free to tweak
    # per_device_train_batch_size=1000, # the training batch size, put it as high as your GPU memory fits
    # gradient_accumulation_steps=8,  # accumulating the gradients before updating the weights
    # per_device_eval_batch_size=1000,  # evaluation batch size
    # logging_steps=500,             # evaluate, log and save model checkpoints every 1000 step
    # save_steps=500,
    load_best_model_at_end=True,  # whether to load the best model (in terms of loss) at the end of training
    save_total_limit=6,           # whether you don't have much space so you let only 3 model weights saved in the disk
    learning_rate=params["lr"],
    weight_decay=params["weight_decay"],
    warmup_steps=10000,
    per_device_train_batch_size=32,  #64
    per_gpu_eval_batch_size=32,  #64
)


trainer = Trainer(
    model=model.to("cuda"),
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
    data_collator=data_collator,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=5),CallbackForNNI],
)




# %%

# trainer.train(resume_from_checkpoint=True)
trainer.train()
nni.report_final_result(loss)
# # %%
# from transformers import pipeline
# unmasker = pipeline('fill-mask', model=model.to("cpu"),tokenizer=tokenizer)
# unmasker("Hello I'm a [MASK] model.")

# # %%


# import math
# model = model.to("cuda")
# eval_results = trainer.evaluate()
# print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")




