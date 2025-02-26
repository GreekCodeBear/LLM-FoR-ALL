import os
import torch
from datasets import load_dataset,Dataset
import wandb
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    AdamW,
)
from utils.utils import find_files,tokenize_dataset

TRUNK_SIZE = 512
TMP_PATH = "/archive/share/cql/aaa/tmp"
DATA_PATH = "data/pt"
OUTPUT_PATH = "results/pt"
CONFIG_PATH = "models/Qwen2.5-0.5B"
WANDB_LOG = True
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'

output_path = OUTPUT_PATH
model_path = CONFIG_PATH
config = AutoConfig.from_pretrained(model_path)
model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
tokenizer = AutoTokenizer.from_pretrained(model_path)

tokenized_datapath = os.path.join(DATA_PATH, "tokenized_dataset")

if not os.path.isdir(tokenized_datapath):
    directories = [
        "film_entertainment",
        "literature_emotion",
        "news_media",
    ]
    data_files = find_files(directories)
    dataset = load_dataset("parquet", data_files=data_files, split="train", columns=["text"], cache_dir=TMP_PATH) 
    dataset = dataset.shuffle(seed=42)
    
    def map_callback(examples):
        result, _ = tokenize_dataset(examples, tokenizer, TRUNK_SIZE)
        return result
    train_dataset = dataset.map(
        map_callback,
        batched=True,
        batch_size=5000,
        remove_columns=dataset.column_names,
        num_proc=32,
    )
    train_dataset.save_to_disk(tokenized_datapath)
    
train_dataset = Dataset.load_from_disk(tokenized_datapath)

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=output_path,
    overwrite_output_dir=True,
    learning_rate=1e-5,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    num_train_epochs=3,
    per_device_train_batch_size=24,
    gradient_accumulation_steps=16,
    save_steps=10_000, 
    save_total_limit=3,
    gradient_checkpointing=True,
    bf16=True,
    logging_steps=10,
    report_to="wandb",
)

if WANDB_LOG:
    wandb.login()
    wandb.init(
        project="qwen-0.5B-pt",name="qwen-0.5B-pt"
    )

# optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collator,
    train_dataset=train_dataset,
    # optimizers=(optimizer, None)  
)
torch.cuda.empty_cache()

# 开始训练
trainer.train()
trainer.save_model()  
tokenizer.save_pretrained(output_path)  