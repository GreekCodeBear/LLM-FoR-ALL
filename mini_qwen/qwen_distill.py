import os
import torch
import wandb
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.distill_utils import DistillConfig, DistillTrainer
from utils.utils import find_files,print_trainable_parameters,format_to_chatml

# 设置环境变量以优化CUDA内存分配
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'

WANDB_LOG = True
TMP_PATH = "/archive/share/cql/aaa/tmp"

output_path = "/archive/share/cql/LLM-FoR-ALL/mini_qwen/results/distill"
model_path = "/archive/share/cql/LLM-FoR-ALL/mini_qwen/data/Qwen2.5-0.5B"
teacher_model_path = "/archive/share/cql/LLM-FoR-ALL/mini_qwen/data/Qwen2.5-0.5B"

# 加载模型
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
tokenizer = AutoTokenizer.from_pretrained(model_path)
print_trainable_parameters(model)

# 加载数据集并进行预处理
directories = ["data"]
data_files = find_files(directories,"/archive/share/cql/LLM-FoR-ALL/mini_qwen/data/distill/numina-deepseek-DeepSeek-R1-Distill-Qwen-7B")
dataset = load_dataset("parquet", data_files=data_files, split="train", columns=["problem", "generation"], cache_dir=TMP_PATH) 
dataset = dataset.shuffle(seed=42)
train_dataset, valid_dataset = dataset.train_test_split(test_size=0.08).values()

# 将训练和验证数据集转换为chatlm格式
train_data = format_to_chatml(train_dataset)
valid_data = format_to_chatml(valid_dataset)
train_dataset = Dataset.from_dict(train_data)
valid_dataset = Dataset.from_dict(valid_data)

# 训练参数配置
training_args = DistillConfig(
    output_dir=output_path,
    overwrite_output_dir=True,
    eval_steps = 1000,
    learning_rate=1e-5,
    warmup_ratio=0.1,
    temperature=0.9,
    alpha = 1,
    max_new_tokens = 1024,
    lr_scheduler_type="cosine",
    num_train_epochs=4,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    save_steps=1000, 
    save_total_limit=3,
    bf16=True,
    logging_steps=10,
    report_to="wandb",
)
if WANDB_LOG:
    wandb.login()
    wandb.init(
        project="qwen-0.5B-deepseek-distill",name="q2.5-0.5B-r1-distill-qwen-7B"
    )

# 初始化Trainer
trainer = DistillTrainer(
    model=model,
    teacher_model=teacher_model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)

# 开始训练
print("Training...")
trainer.train()
trainer.save_model()  
tokenizer.save_pretrained(output_path) 