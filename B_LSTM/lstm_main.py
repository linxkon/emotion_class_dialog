import pandas as pd
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter  #使用tensorboard可视化
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import jieba
import numpy as np
from pathlib import Path
from lstm_model import *


#主要训练流程
def main():
    path_data=Path('Data') / 'ECG_data.csv'
    path_model=Path('B_LSTM/model') / 'lstm_model.pth'

    #超参数
    num_epochs = 30
    batch_size=32
    max_len = 128
    learning_rate = 8e-3
    dropout = 0.5

    # 读取数据
    df = pd.read_csv(path_data)
    
    # 提取问答对和标签
    texts = df['sentence'].str.split('xxx：').str[-1].tolist()  # 筛选处理范围
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
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
    # breakpoint()
    # 创建数据集实例
    train_dataset = TextDataset(X_train, y_train, vocab, label2idx)
    test_dataset = TextDataset(X_test, y_test, vocab, label2idx)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size)
    
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
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练模型
    best_accuracy = 0
    writer = SummaryWriter()

    for epoch in range(num_epochs):

        epoch_loss = train_model(model, train_loader, criterion, optimizer, device)
        # 评估模型
        train_acc = evaluate_model(model, train_loader, device)
        print(f'Train Accuracy: {train_acc:.4f}')        
        writer.add_scalar('loss/Train', epoch_loss, epoch*batch_size)
        writer.add_scalar('ACC/Train', train_acc, epoch*batch_size)

        val_acc = evaluate_model(model, test_loader, device)
        writer.add_scalar('ACC/Validation', val_acc, epoch*batch_size)
        print(f'Test Accuracy: {val_acc:.4f}')
        
        # 保存最佳模型
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            model_best = model
        # print(f'Epoch: {epoch+1}, Average Loss: {total_loss/len(train_loader):.4f}')

    
    # 保存模型
    torch.save({
        'model_state_dict': model_best.state_dict(),
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