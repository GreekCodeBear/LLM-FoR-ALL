import os
import torch
import wandb
from peft import LoraConfig
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from utils.utils import find_files,formatting_prompts_func,print_trainable_parameters

# 设置环境变量以优化CUDA内存分配
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'

WANDB_LOG = True
TMP_PATH = "/archive/share/cql/aaa/tmp"

output_path = "/archive/share/cql/LLM-FoR-ALL/mini_qwen/results/sft-1"
model_path = "/archive/share/cql/LLM-FoR-ALL/mini_qwen/results/sft/checkpoint-6000"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
tokenizer = AutoTokenizer.from_pretrained(model_path)
print_trainable_parameters(model)

# 加载数据集并进行预处理
directories = ["Gen","7M"]
data_files = find_files(directories,"/archive/share/cql/LLM-FoR-ALL/mini_qwen/data/sft")
dataset = load_dataset("parquet", data_files=data_files, split="train", columns=["conversations"], cache_dir=TMP_PATH) 
dataset = dataset.shuffle(seed=42)
train_dataset, valid_dataset = dataset.train_test_split(test_size=0.2).values()

# 数据整理器
response_template = "<|im_start|>assistant\n"
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]
collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer, mlm=False)

# 训练LoRA
# lora_config = LoraConfig(
#     r=8,
#     lora_alpha=16,
#     target_modules=["q", "v"],
#     lora_dropout=0.01,
#     bias="none",
#     task_type="CAUSAL_LM",
# )

# 训练参数配置
training_args = SFTConfig(
    output_dir=output_path,
    overwrite_output_dir=True,
    eval_steps = 2000,
    learning_rate=1e-5,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    num_train_epochs=3,
    per_device_train_batch_size=12,
    gradient_accumulation_steps=16,
    save_steps=1000,  # 保存中间模型
    save_total_limit=3,
    bf16=True,
    logging_steps=10,
    report_to="wandb",
)

if WANDB_LOG:
    wandb.login()
    wandb.init(
        project="qwen-0.5B-sft",name="qwen-0.5B-sft"
    )


# 初始化Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    # peft_config=lora_config,
    args=training_args,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    max_seq_length=512,
    packing=False,
    dataset_num_proc=16,
    dataset_batch_size=5000,
)

# 开始训练
print("Training...")
trainer.train()
trainer.save_model()  
tokenizer.save_pretrained(output_path) 