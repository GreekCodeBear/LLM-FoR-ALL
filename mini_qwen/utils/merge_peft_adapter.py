import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

AdapterModelPath = "results/sft/checkpoint-5000"
BaseModelPath = "results/sft/checkpoint-5000"
OutputPath = "results/sft/final_model"

model = AutoModelForCausalLM.from_pretrained(
    BaseModelPath, return_dict=True, torch_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(BaseModelPath)

# Load the PEFT model
model = PeftModel.from_pretrained(model, AdapterModelPath)
model.eval()

model = model.merge_and_unload()

model.save_pretrained(f"{OutputPath}")
tokenizer.save_pretrained(f"{OutputPath}")
# 上传 HF
# model.push_to_hub(f"{OutputPath}", use_temp_dir=False)