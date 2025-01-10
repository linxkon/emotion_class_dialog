from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import accuracy_score
from pathlib import Path
import joblib

# 指定数据集的位置
path_data_train=Path('A_random_forest/rf_data') / 'ECG_data_rf.csv'
path_data_stopwords=Path('A_random_forest') / 'stopwords.txt'

# 读取数据集
content = pd.read_csv(path_data_train)
# 获得分词结果
WORDS_COLUMN = 'words'
corpus = content[WORDS_COLUMN].values
# print('corpus:',corpus[:2])

# 读取停用词
stop_words = open(path_data_stopwords).read().split()
# print('stop_words:',stop_words[:2])

# 特征工程——计算tfidf特征
tfidf = TfidfVectorizer(stop_words=stop_words)
text_vectors = tfidf.fit_transform(corpus)


# 划分数据集
targets = content['label'] # 目标值
x_train, x_test, y_train, y_test = train_test_split(text_vectors, targets, test_size=0.2, random_state=0)

# 实例化模型
model = RandomForestClassifier(random_state=42,n_estimators=100)
# 模型训练
model.fit(x_train, y_train)
# 模型评估
accuracy_train = accuracy_score(model.predict(x_train), y_train)
accuracy_test = accuracy_score(model.predict(x_test), y_test)
print('模型准确率:')
print('Accuracy train:',accuracy_train)
print('Accuracy test:',accuracy_test)

path_model=Path('A_random_forest/model') / 'random_forest_model.pkl'
path_vectorizer=Path('A_random_forest/model') / 'tfidf_vectorizer.pkl'
# 保存模型
joblib.dump(model, path_model)

# 保存矢量化器
joblib.dump(tfidf, path_vectorizer)

print("模型和矢量化器已保存。")