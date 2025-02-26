# 下载预训练数据集 5.3B tokens
modelscope download --dataset 'BAAI/IndustryCorpus2' --include 'film_entertainment/*/high*' --local_dir 'data/pt' # 数据量较大，英文文件选择前3个文件
find 'data/pt/film_entertainment/english/high' -maxdepth 1 -type f | sort | tail -n +4 | xargs rm -f
modelscope download --dataset 'BAAI/IndustryCorpus2' --include 'news_media/*/high*' --local_dir 'data/pt'
modelscope download --dataset 'BAAI/IndustryCorpus2' --include 'literature_emotion/*/high*' --local_dir 'data/pt' # 数据量较大，英文文件选择前3个文件
find 'data/pt/literature_emotion/english/high' -maxdepth 1 -type f | sort | tail -n +4 | xargs rm -f

# 下载预训练数据集MMLU
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar
mkdir -p data/mmlu
mv data.tar data/mmlu
tar xf data/mmlu/data.tar -C data/mmlu

# 下载指令微调训练数据集
modelscope download --dataset 'BAAI/Infinity-Instruct' --local_dir 'data/sft' # 选择7M和Gen进行微调，因为这两个数据集更新时间最近，且数据量大

# 下载偏好数据集
modelscope download --dataset 'BAAI/Infinity-Preference' --local_dir 'data/dpo'

# 下载蒸馏数据集
modelscope download --dataset HuggingFaceH4/numina-deepseek-r1-qwen-7b --local_dir 'data/distill'

# 下载偏好数据集
modelscope download --dataset swift/stack-exchange-paired --local_dir 'data/reward'