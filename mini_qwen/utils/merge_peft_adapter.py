import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

AdapterModelPath = "/archive/share/cql/LLM-FoR-ALL/mini_qwen/results/rm/Qwen2.5-0.5B_peft_stack-exchange-paired__0_2e-05/checkpoint-1908"
BaseModelPath = "/archive/share/cql/LLM-FoR-ALL/mini_qwen/data/Qwen2.5-0.5B"
OutputPath = "results/rm/final_model"

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