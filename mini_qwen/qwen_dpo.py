import os
import torch
import wandb
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
from utils.utils import find_files

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'

WANDB_LOG = True
TMP_PATH = "/archive/share/cql/aaa/tmp"
REPO_PATH = "/archive/share/cql/LLM-FoR-ALL/mini_qwen"

output_path = "results/dpo"
data_path = "data/dpo"
model_path = "results/sft-1/checkpoint-14000"  
model_path = os.path.join(REPO_PATH, model_path)
output_path = os.path.join(REPO_PATH, output_path)
data_path = os.path.join(REPO_PATH, data_path)

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
tokenizer = AutoTokenizer.from_pretrained(model_path)

data_files = ["/archive/share/cql/LLM-FoR-ALL/mini_qwen/data/dpo/test-00000-of-00001.parquet","/archive/share/cql/LLM-FoR-ALL/mini_qwen/data/dpo/train-00000-of-00001.parquet"]
dataset = load_dataset("parquet", data_files=data_files, split="train")
dataset = dataset.shuffle(seed=42)

def preprocess_dataset(examples):
    prompt, chosen, rejected = [], [], []
    for i in range(len(examples["prompt"])):
        text = f"<|im_start|>user\n{examples['prompt'][i]}<|im_end|>\n<|im_start|>assistant\n"
        prompt.append(text)

        assert examples["chosen"][i][1]["role"] == "assistant"
        text = f"{examples['chosen'][i][1]['content']}<|im_end|>"
        chosen.append(text)

        assert examples["rejected"][i][1]["role"] == "assistant"
        text = f"{examples['rejected'][i][1]['content']}<|im_end|>"
        rejected.append(text)

    result = {"prompt": prompt, "chosen": chosen, "rejected": rejected}
    return result

train_dataset = dataset.map(
    preprocess_dataset,
    batched=True,
    batch_size=5000,
    remove_columns=dataset.column_names,
    num_proc=16,
)

training_args = DPOConfig(
    output_dir=output_path,
    overwrite_output_dir=True,
    learning_rate=5e-7,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    save_steps=500,  # 保存中间模型
    save_total_limit=10,
    bf16=True,
    logging_steps=10,
    report_to="wandb",
)

if WANDB_LOG:
    wandb.login()
    wandb.init(
        project="qwen-0.5B-dpo",name="qwen-0.5B-dpo"
    )

trainer = DPOTrainer(
    model=model,
    train_dataset=train_dataset,
    args=training_args,
    tokenizer=tokenizer,
    dataset_num_proc=16,
    max_length=1024,
    max_prompt_length=512,
)

trainer.train()
trainer.save_model() 
tokenizer.save_pretrained(output_path) 