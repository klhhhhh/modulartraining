# %%
from transformers import BertLayer,BertConfig,BertModel,BertForMaskedLM
from transformers import BertForMaskedLM
from transformers import BertConfig
from transformers import BertTokenizer
import datasets
import json
import sys
import copy
from datasets import load_dataset
BertBaseconfig = BertConfig()
BertBase = BertForMaskedLM(BertBaseconfig)
layer = BertLayer(BertBaseconfig)
MetaModel = BertForMaskedLM.from_pretrained('/home/wanzhipeng/deepincubation/MetaModel_bert_wiki/checkpoint-36500')
MetaModelEncoderBertLayer = MetaModel.bert.encoder.layer
BaseModelEncoderBertLayer = BertBase.bert.encoder.layer
BaseLayerNums = BaseModelEncoderBertLayer.__len__()
MetaLayerNums = MetaModelEncoderBertLayer.__len__()
Submodules = []



# 用子模块替换对应的元模型
def initSubmodules():
    global Submodules
    Submodules = []
    scale = BaseLayerNums // MetaLayerNums   
    for i in range(MetaLayerNums):
        layers = [BertLayer(BertBaseconfig) for _ in range(scale)]
        Submodule = copy.deepcopy(MetaModel)
        Submodule.bert.encoder.layer = Submodule.bert.encoder.layer[0:i+1] + layers + Submodule.bert.encoder.layer[i+1:]
        del Submodule.bert.encoder.layer[i]
        Submodules.append(Submodule)


def tokenize_function(examples):
    return tokenizer(examples["text"])
initSubmodules()
meta_layer_index = 3
model=Submodules[meta_layer_index]
model.config.num_hidden_layers = 6
layer_nums = model.config.num_hidden_layers
model.bert.encoder.meta_layer_index = meta_layer_index
model.bert.encoder.scale = 3

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',use_fast=True) 
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
training_args = TrainingArguments(
    output_dir="sub4",          # output directory to where save model checkpoint
    evaluation_strategy="steps",    # evaluate each `logging_steps` steps
    logging_strategy="steps",
    overwrite_output_dir=True,      
    num_train_epochs=100,            # number of training epochs, feel free to tweak
    logging_steps=500,             # evaluate, log and save model checkpoints every 1000 step
    save_steps=500,
    load_best_model_at_end=True,  # whether to load the best model (in terms of loss) at the end of training
    save_total_limit=3,           # whether you don't have much space so you let only 3 model weights saved in the disk
    learning_rate=1e-5,
    weight_decay=0.01,
    warmup_steps=10000,
    per_device_train_batch_size=64,
    per_gpu_eval_batch_size=64,
)
from transformers import Trainer, TrainingArguments,EarlyStoppingCallback,TrainerCallback
from transformers import DataCollatorForLanguageModeling
import torch

## 不同卡的情况下会出问题
class CallbackForMetaLayer(TrainerCallback):
    
    def __init__(self):
        super().__init__()
        self.meta_layer_version = 1
        self.other_mata_layer_version = [1] * layer_nums

    # def on_step_begin(self, args, state, control, model=None, **kwargs):
    #     self.meta_layer_outputs = [model.bert.encoder.meta_layer_outputs]
    #     print("step_begin:")
    #     print(id(self.meta_layer_outputs[0]))
           
        
    
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        p = 0.7
        torch.save(model.bert.encoder.meta_layer.state_dict(), f'test2/meta_layer_{meta_layer_index}_version_{self.meta_layer_version}_params.pt')
        self.meta_layer_version += 1
        for i in range(layer_nums):
            if i in range(meta_layer_index, meta_layer_index+3):
                print(f"jump:{i}")
                continue
            else:
                try:
                    loaded_meta_layer_params = torch.load(f'test2/meta_layer_{i}_version_{self.other_mata_layer_version[i]}_params.pt')
                except FileNotFoundError:
                    loaded_meta_layer_params = None
                if loaded_meta_layer_params is not None:
                    self.other_mata_layer_version[i] += 1
                    latest_meta_layer = BertLayer(BertBaseconfig)
                    latest_meta_layer.load_state_dict(loaded_meta_layer_params, strict=True)
                    for param_a, param_b in zip(latest_meta_layer.parameters(), model.bert.encoder.layer[i].parameters()):
                        param_a.data = p * param_a.data + (1-p) * param_b.data.to(param_a.device)
                else:
                    print(f"None:{i}")
 
        

        
                          
trainer = Trainer(
    model=model.to("cuda"),
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
    data_collator=data_collator,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=5), CallbackForMetaLayer],
)        

# trainer.train(resume_from_checkpoint=True)
trainer.train()

