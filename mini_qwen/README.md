# Mini Qwen

## Download Data

'''
bash download_data.sh
'''

## Pretrain

- single GPU
'''
CUDA_VISIBLE_DEVICES=0 python mini_qwen_pt.py
'''

- multi GPU
'''
accelerate launch --config_file accelerate_config.yaml qwen_pt.py

nohup accelerate launch --config_file accelerate_config.yaml qwen_pt.py > logs/output_pt.log 2>&1 &
'''

> 混合精度训练: 全精度加载模型，并且traning_args.fp16 = True, 即使用 `--fp16` 参数
> 单精度训练: 全精度加载模型，并且traning_args.fp16 = False, 即不使用 `--fp16` 参数，也不使用 `--bf16` 参数

## SFT



## ChatBot
'''
python qwen_chat.py
'''

## Evaluation

'''
python qwen_eval.py --checkpoint-path <model_path> --eval_data_path <data_path>
'''