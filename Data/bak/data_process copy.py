import pandas as pd
from collections import Counter
import numpy as np
import jieba
from pathlib import Path

# 获取data目录
path_data_index=Path('Data') / 'index.csv'
path_data_train=Path('Data') / 'ECG_train_split_train.csv'
path_data_test=Path('Data') / 'ECG_train_split_test.csv'

content_index = pd.read_csv(path_data_index, sep=',')
content_train = pd.read_csv(path_data_train, sep=',')
content_test = pd.read_csv(path_data_test, sep=',')


# 打印前10行
print('标签:',content_index.head(10))
print('训练集',content_train.head(10))
# print('测试集',content_test.head(10))