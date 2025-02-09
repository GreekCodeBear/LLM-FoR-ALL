import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

# 加载分词器与模型 
model_path = "/archive/share/cql/LLM-FoR-ALL/mini_qwen/results/pt/checkpoint-10000"
# model_path = "results/sft"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16, #确保与训练一致
    attn_implementation="flash_attention_2",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_path)


while True:
    prompt = input("user：")
    
    text = prompt  # 预训练模型
    # text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"  # 微调和直接偏好优化模型
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print("qwen：", response)