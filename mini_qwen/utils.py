import os
from itertools import chain

def find_files(dirs):
    files = []
    for dir in dirs:
        base_path = os.path.join("data/pt", dir)
        for dirpath, _, filenames in os.walk(base_path):
            for filename in filenames:
                if filename.endswith(".parquet"):
                    full_path = os.path.join(dirpath, filename)
                    files.append(full_path)
    return files

def tokenize_dataset(examples,tokenizer,block_size=512):
    """预处理预训练数据集，将文本分词并分块"""
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