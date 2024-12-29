import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import jieba
import numpy as np
from pathlib import Path
from lstm import *


#主要训练流程
def main():
    path_data=Path('Data') / 'ECG_data.csv'
    path_model=Path('B_LSTM/model') / 'lstm_model.pth'
    # 读取数据
    df = pd.read_csv(path_data)
    
    # 提取问答对和标签
    texts = df['sentence'].str.split('AAA：').str[-1].tolist()  # 只取回答部分
    labels = df['label'].tolist()
    
    # 构建词汇表
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for text in texts:
        for token in jieba.cut(text):
            if token not in vocab:
                vocab[token] = len(vocab)
    
    # 构建标签映射
    unique_labels = list(set(labels))
    label2idx = {label: idx for idx, label in enumerate(unique_labels)}
    
    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.1, random_state=42)
    # breakpoint()
    # 创建数据集实例
    train_dataset = TextDataset(X_train, y_train, vocab, label2idx)
    test_dataset = TextDataset(X_test, y_test, vocab, label2idx)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TextClassifier(
        vocab_size=len(vocab),
        embedding_dim=100,
        hidden_dim=128,
        num_classes=len(unique_labels)
    ).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.008)
    
    # 训练模型
    train_model(model, train_loader, criterion, optimizer, device,test_loader,num_epochs=30)
    
    # 评估模型
    accuracy = evaluate_model(model, train_loader, device)
    print(f'Train Accuracy: {accuracy:.4f}')

    accuracy = evaluate_model(model, test_loader, device)
    print(f'Test Accuracy: {accuracy:.4f}')
    
    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'label2idx': label2idx,
        'model_config': {
            'vocab_size': len(vocab),
            'embedding_dim': 100,
            'hidden_dim': 128,
            'num_classes': len(unique_labels)
        }
    }, path_model)
    print(f'Model saved to {path_model}')

if __name__ == '__main__':
    main()