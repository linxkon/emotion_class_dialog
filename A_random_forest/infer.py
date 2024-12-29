import joblib
from pathlib import Path

path_model=Path('A_random_forest/model') / 'random_forest_model.pkl'
path_vectorizer=Path('A_random_forest/model') / 'tfidf_vectorizer.pkl'

# 加载模型和矢量化器
loaded_model = joblib.load(path_model)
loaded_tfidf = joblib.load(path_vectorizer)

# 示例：对新文本数据进行预测
new_texts = ["Q:快春节啦！    A：嘿嘿…"]
new_vectors = loaded_tfidf.transform(new_texts)
predictions = loaded_model.predict(new_vectors)

print("预测结果:", predictions)
