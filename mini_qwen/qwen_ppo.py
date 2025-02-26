import torch
import wandb
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import Adafactor, AutoTokenizer, pipeline, set_seed

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from trl.core import LengthSampler
from utils.utils import find_files,preprocess_ppo_dataset,collator_ppo
tqdm.pandas()

WANDB_LOG = True
NUM_EPOCH = 10
MODEL_PATH = "/archive/share/cql/LLM-FoR-ALL/mini_qwen/data/Qwen2.5-0.5B"
REWARD_PATH = "/archive/share/cql/LLM-FoR-ALL/mini_qwen/results/rm/final_model"
OUTPUT_PATH = "/archive/share/cql/LLM-FoR-ALL/mini_qwen/results/ppo"
TMP_PATH = "/archive/share/cql/aaa/tmp"
SEED = 42
MAX_LENGTH = 512
GRADIENT_CHECKPOINTING = True
LR = 1.41e-5
STEPS = 20000
BS = 8
TRAIN_SUBSET = 0
NUM_PROC = 16

set_seed(SEED)
if WANDB_LOG:
    wandb.login()
    wandb.init(
        project="qwen-0.5B-ppo",name="qwen-0.5B-ppo"
    )

data_files = find_files(["rl"],"/archive/share/cql/LLM-FoR-ALL/mini_qwen/data/reward/data")
train_dataset = load_dataset("parquet", data_files=data_files, split="train", cache_dir=TMP_PATH) 
original_columns = train_dataset.column_names
if TRAIN_SUBSET>0:
    train_dataset = train_dataset.select(range(TRAIN_SUBSET))

# load tokenizer for dataset process
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
# load RM tokenizer for dataset process
rm_tokenizer = AutoTokenizer.from_pretrained(REWARD_PATH)
if rm_tokenizer.pad_token is None:
    rm_tokenizer.pad_token = rm_tokenizer.eos_token

# process dataset
dataset = train_dataset.map(
    lambda examples: preprocess_ppo_dataset(examples, tokenizer),
    batched=True,
    num_proc=NUM_PROC,
    remove_columns=original_columns,
)
dataset = dataset.filter(lambda x: len(x["input_ids"]) < 512, batched=False, num_proc=NUM_PROC)
dataset.set_format(type="torch")

# ppo config
current_device = Accelerator().local_process_index
config = PPOConfig(
    steps=STEPS,
    model_name=MODEL_PATH,
    learning_rate=LR,
    log_with='wandb',
    batch_size=BS*4,
    mini_batch_size=BS,
    gradient_accumulation_steps=4,
    optimize_cuda_cache=True,
    target_kl=0.1,
    ppo_epochs=3,
    seed=SEED,
    init_kl_coef=0.2,
    adap_kl_ctrl=True,
)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Define Policy Trainer
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name,
    device_map={"": current_device},
    peft_config=lora_config,
)
optimizer = Adafactor(
    filter(lambda p: p.requires_grad, model.parameters()),
    scale_parameter=False,
    relative_step=False,
    warmup_init=False,
    lr=config.learning_rate,
)
ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator_ppo,
    optimizer=optimizer,
)

# init Reward Model, set the device to the same device as the PPOTrainer.
sent_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": 16,
    "truncation": True,
}
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model=REWARD_PATH,
    device_map={"": current_device},
    tokenizer=rm_tokenizer,
    return_token_type_ids=False,
)
if sentiment_pipe.model.config.pad_token_id is None:
    sentiment_pipe.model.config.pad_token_id = sentiment_pipe.model.config.eos_token_id

# config for policy model generation
generation_kwargs = {
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": 100_000,
}
output_min_length = 32
output_max_length = 128
output_length_sampler = LengthSampler(output_min_length, output_max_length)

for epoch in tqdm(range(NUM_EPOCH), "epoch: "):
    if epoch >= config.total_ppo_epochs:
        break
    for batch in tqdm(ppo_trainer.dataloader): 
        question_tensors = batch["input_ids"]
    
        response_tensors = ppo_trainer.generate(
            question_tensors,
            return_prompt=False,
            length_sampler=output_length_sampler,
            **generation_kwargs,
        )
        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

        # Compute reward score (using the sentiment analysis pipeline)
        texts = [q+r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
        rewards = [torch.tensor(output[0]["score"]) for output in pipe_outputs]

        # Run PPO step
        stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)
        
    if epoch and epoch % 1 == 0:
        ppo_trainer.save_pretrained(OUTPUT_PATH + f"step_{epoch}")

#### Save model
ppo_trainer.save_pretrained(OUTPUT_PATH)