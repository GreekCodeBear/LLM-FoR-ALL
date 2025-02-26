import os
from itertools import chain

def collator_ppo(data):
    # for PPO use
    return {key: [d[key] for d in data] for key in data[0]}

def preprocess_ppo_dataset(examples,tokenizer):
    # for PPO use
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        for question in examples["question"]:
            query = "Question: " + question + "\n\nAnswer: "
            tokenized_question = tokenizer(query, truncation=True)
            new_examples["query"].append(query)
            new_examples["input_ids"].append(tokenized_question["input_ids"])

        return new_examples

def preprocess_rm_dataset(examples,tokenizer):
    # for RM use
    # Turn the dataset into pairs of post + summaries, where text_j is the preferred question + answer and text_k is the other.
    new_examples = {
        "input_ids_j": [],
        "attention_mask_j": [],
        "input_ids_k": [],
        "attention_mask_k": [],
    }
    for question, response_j, response_k in zip(examples["question"], examples["response_j"], examples["response_k"]):
        tokenized_j = tokenizer("Question: " + question + "\n\nAnswer: " + response_j, truncation=True)
        tokenized_k = tokenizer("Question: " + question + "\n\nAnswer: " + response_k, truncation=True)

        new_examples["input_ids_j"].append(tokenized_j["input_ids"])
        new_examples["attention_mask_j"].append(tokenized_j["attention_mask"])
        new_examples["input_ids_k"].append(tokenized_k["input_ids"])
        new_examples["attention_mask_k"].append(tokenized_k["attention_mask"])

    return new_examples

def format_to_chatml(data):
    formatted_data = []
    for sample in data:
        problem = sample["problem"]
        generation = sample["generation"]
        
        formatted_data.append([
                {"role": "user", "content": problem},
                {"role": "assistant", "content": generation}
            ]
        )
    return {"messages": formatted_data}

def formatting_prompts_func_distill(example):
    # for distill use
    output_texts = []
    for i in range(len(example["problem"])):
        human_text = example["problem"][i]
        gpt_text = example["generation"][i]
        text = f"<|im_start|>user\n{human_text}<|im_end|>\n<|im_start|>assistant\n{gpt_text}<|im_end|>"
        output_texts.append(text)
    return output_texts

def formatting_prompts_func(example):
    # for sft use
    output_texts = []
    for i in range(len(example["conversations"])):
        for item in example["conversations"][i]:
            if item["from"] == "human":
                human_text = item["value"]
            elif item["from"] == "gpt":
                gpt_text = item["value"]
            else:
                raise ValueError(f"Unknown sender: {item['from']}")
        text = f"<|im_start|>user\n{human_text}<|im_end|>\n<|im_start|>assistant\n{gpt_text}<|im_end|>"
        output_texts.append(text)
    return output_texts

def find_files(dirs,path="data/pt"):
    """
    遍历目录，查找所有文件
    """
    files = []
    for dir in dirs:
        base_path = os.path.join(path, dir)
        for dirpath, _, filenames in os.walk(base_path):
            for filename in filenames:
                if filename.endswith(".parquet"):
                    full_path = os.path.join(dirpath, filename)
                    files.append(full_path)
    return files

def tokenize_dataset(examples,tokenizer,block_size=512):
    """
    预处理预训练数据集，将文本分词并分块
    """
    eos_token = "<|im_end|>"
    text_examples = [text + eos_token for text in examples["text"]]  # 添加结束符
    tokenized_examples = tokenizer(text_examples, add_special_tokens=False)

    concatenated_examples = {
        k: list(chain(*tokenized_examples[k])) for k in tokenized_examples.keys()
    }
    
    total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
    total_length = (total_length // block_size) * block_size  # 对齐块大小

    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result,total_length

def print_trainable_parameters(model):
    """
    打印模型中可训练参数的数量。
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    
