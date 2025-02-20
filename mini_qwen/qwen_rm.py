import evaluate
import numpy as np
import wandb
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from utils.rm_utils import RewardDataCollatorWithPadding, RewardTrainer
from utils.utils import find_files,preprocess_rm_dataset

WANDB_LOG = True
MODEL_PATH = "/archive/share/cql/LLM-FoR-ALL/mini_qwen/data/Qwen2.5-0.5B"
OUTPUT_PATH = "/archive/share/cql/LLM-FoR-ALL/mini_qwen/results/rm"
TMP_PATH = "/archive/share/cql/aaa/tmp"
SEED = 42
MAX_LENGTH = 512
GRADIENT_CHECKPOINTING = True
LR = 2e-5
BS = 32
TRAIN_SUBSET = 0
EVAL_SUBSET = 0
NUM_PROC = 16

set_seed(SEED)

data_files = find_files(["reward"],"/archive/share/cql/LLM-FoR-ALL/mini_qwen/data/reward/data")
train_dataset = load_dataset("parquet", data_files=data_files, split="train", cache_dir=TMP_PATH) 
train_dataset = train_dataset.shuffle(seed=42)

data_files = find_files(["evaluation"],"/archive/share/cql/LLM-FoR-ALL/mini_qwen/data/reward/data")
eval_dataset = load_dataset("parquet", data_files=data_files, split="train", cache_dir=TMP_PATH) 
eval_dataset = eval_dataset.shuffle(seed=42)

# 看是否需要截断数据集（用于测试）
if TRAIN_SUBSET > 0:
    train_dataset = train_dataset.select(range(TRAIN_SUBSET))
if EVAL_SUBSET > 0:
    eval_dataset = eval_dataset.select(range(EVAL_SUBSET))

# 加载 peft SEQ_CLS model 
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH, num_labels=1, torch_dtype=torch.bfloat16
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_auth_token=True)
tokenizer.pad_token = tokenizer.eos_token

# 为模型添加padding token.
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id
model.config.use_cache = not GRADIENT_CHECKPOINTING
original_columns = train_dataset.column_names

# 预处理数据集，过滤超出max_length的QAs
train_dataset = train_dataset.map(
    lambda examples: preprocess_rm_dataset(examples, tokenizer),
    batched=True,
    num_proc=NUM_PROC,
    remove_columns=original_columns,)
train_dataset = train_dataset.filter(
    lambda x: len(x["input_ids_j"]) <= MAX_LENGTH and len(x["input_ids_k"]) <= MAX_LENGTH,
    num_proc=NUM_PROC,)
eval_dataset = eval_dataset.map(
    lambda examples: preprocess_rm_dataset(examples, tokenizer),
    batched=True,
    num_proc=NUM_PROC,
    remove_columns=original_columns,)
eval_dataset = eval_dataset.filter(
    lambda x: len(x["input_ids_j"]) <= MAX_LENGTH and len(x["input_ids_k"]) <= MAX_LENGTH,
    num_proc=NUM_PROC,)
    
# 定义validation metric
accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions, _ = eval_pred
    predictions = np.argmax(predictions, axis=0)
    labels = np.zeros(predictions.shape)
    return accuracy.compute(predictions=predictions, references=labels)
    
# 训练参数设置
training_args = TrainingArguments(
    output_dir=f"/archive/share/cql/LLM-FoR-ALL/mini_qwen/results/rm/{MODEL_PATH.split('/')[-1]}_peft_stack-exchange-paired__{TRAIN_SUBSET}_{LR}",
    learning_rate=LR,
    per_device_train_batch_size=BS,
    per_device_eval_batch_size=BS,
    num_train_epochs=3,
    weight_decay=0.001,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    gradient_accumulation_steps=16,
    gradient_checkpointing=GRADIENT_CHECKPOINTING,
    remove_unused_columns=False,
    label_names=[],
    bf16=True,
    logging_strategy="steps",
    logging_steps=10,
    optim="adamw_hf",
    lr_scheduler_type="linear",
    seed=SEED,
    report_to="wandb",
)

if WANDB_LOG:
    wandb.login()
    wandb.init(
        project="qwen-0.5B-rm",name="qwen-0.5B-rm"
    )

trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer),
)

trainer.train()
model.save_pretrained(OUTPUT_PATH)