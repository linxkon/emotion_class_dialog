import pandas as pd
from collections import Counter
import numpy as np
import jieba
from pathlib import Path

# 获取data目录
path_data_index=Path('Data') / 'index.csv'
path_data_ECG=Path('Data') / 'ECG_data.csv'

content_index = pd.read_csv(path_data_index, sep=',')
content_data_ECG = pd.read_csv(path_data_ECG, sep=',')


# 打印前10行
print('标签:',content_index.head(10))
print('数据集',content_data_ECG.head(10))
print('数据集长度:',len(content_data_ECG))