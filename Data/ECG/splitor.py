import pandas as pd
from sklearn.model_selection import train_test_split

# 读取数据
file_path = '/home/home/linxkon/AI_project/emotion_class/ECG_train.csv'
data = pd.read_csv(file_path, header=None, names=['Sample', 'Label'])

# 划分训练集和测试集，确保类别比例一致
X = data['Sample']  # 样本
y = data['Label']   # 标签

# 使用 stratify 确保类别比例一致
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

# 创建训练集和测试集 DataFrame
train_data = pd.DataFrame({'Sample': X_train, 'Label': y_train})
test_data = pd.DataFrame({'Sample': X_test, 'Label': y_test})

# 保存到文件
train_data.to_csv('/home/home/linxkon/AI_project/emotion_class/ECG_train_split_train.csv', index=False)
test_data.to_csv('/home/home/linxkon/AI_project/emotion_class/ECG_train_split_test.csv', index=False)

print("训练集和测试集已划分并保存。")