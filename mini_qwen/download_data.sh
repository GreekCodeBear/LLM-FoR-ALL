# 下载预训练数据集 5.3B tokens
modelscope download --dataset 'BAAI/IndustryCorpus2' --include 'film_entertainment/*/high*' --local_dir 'data/pt' # 数据量较大，英文文件选择前3个文件
find 'data/pt/film_entertainment/english/high' -maxdepth 1 -type f | sort | tail -n +4 | xargs rm -f

modelscope download --dataset 'BAAI/IndustryCorpus2' --include 'news_media/*/high*' --local_dir 'data/pt'
modelscope download --dataset 'BAAI/IndustryCorpus2' --include 'literature_emotion/*/high*' --local_dir 'data/pt' # 数据量较大，英文文件选择前3个文件
find 'data/pt/literature_emotion/english/high' -maxdepth 1 -type f | sort | tail -n +4 | xargs rm -f
