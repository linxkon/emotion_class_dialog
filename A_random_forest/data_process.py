import pandas as pd
from collections import Counter
import numpy as np
import jieba
from pathlib import Path
from sklearn.model_selection import train_test_split

# 获取data目录
path_data_index=Path('Data') / 'index.csv'
path_data_ECG=Path('Data') / 'ECG_data.csv'

content_data_ECG = pd.read_csv(path_data_ECG, sep=',')


# 1.数据分析
print('-'*10,'数据集分析','-'*10)
print('数据集',content_data_ECG.head(10))
print('数据集长度:',len(content_data_ECG))
# 统计每个类别的数量
print('-'*10,'统计每个类别的数量','-'*10)
count = Counter(content_data_ECG.label.values)
print(count)

# 2.标签字符处理为数字
print('-'*10,'标签字符处理为数字','-'*10)
content_index = pd.read_csv(path_data_index, sep=',')
print('标签:',content_index)

def convert_labels_to_index(df, label_mapping_df):
    """
    将文本标签转换为数字索引
    
    参数:
    df: 包含文本标签的DataFrame
    label_mapping_df: 从文件读取的标签映射DataFrame
    
    返回:
    处理后的DataFrame副本
    """
    # 创建标签到索引的字典
    label_to_index = dict(zip(label_mapping_df['标签名称'], label_mapping_df.index))
    
    # 创建DataFrame的副本
    processed_df = df.copy()
    
    # 将label列转换为对应的索引
    processed_df['label'] = processed_df['label'].map(label_to_index)
    
    return processed_df

content=convert_labels_to_index(content_data_ECG, content_index)
print(content)

#3.特征工程

# 结巴分词
def cut_sentence(s):
    return list(jieba.cut(s))

# 添加一列数据存储分词后的结果
content['words'] = content['sentence'].apply(cut_sentence)
# 打印前十行
print(content.head(10))
# 分词
content['words'] = content['sentence'].apply(lambda s: ' '.join(cut_sentence(s)))

# 将分词后的结果只保留30个元素
content['words'] = content['words'].apply(lambda s: ' '.join(s.split())[:90])

# 4结果存放到csv文件中
path_data_train=Path('A_random_forest/rf_data') / 'ECG_data_rf.csv'
content.to_csv(path_data_train, index=False)



# # 5.划分训练集和测试集，确保类别比例一致
# X = content['words']  # 样本
# y = content['label']   # 标签

# # 使用 stratify 确保类别比例一致
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

# # 创建训练集和测试集 DataFrame
# train_data = pd.DataFrame({'Sample': X_train, 'Label': y_train})
# test_data = pd.DataFrame({'Sample': X_test, 'Label': y_test})

# # 结果存放到csv文件中
# path_data_train=Path('A_random_forest/rf_data') / 'ECG_data_train.csv'
# path_data_test=Path('A_random_forest/rf_data') / 'ECG_data_test.csv'
# train_data.to_csv(path_data_train, index=False)
# test_data.to_csv(path_data_test, index=False)


