import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import jieba

# 1. 数据预处理类
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, label2idx, max_len=40):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.label2idx = label2idx
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 分词
        tokens = list(jieba.cut(text))
        # 转换为索引
        indexed = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        # 处理长度
        if len(indexed) > self.max_len:
            indexed = indexed[:self.max_len]
        else:
            indexed = indexed + [self.vocab['<PAD>']] * (self.max_len - len(indexed))
            
        return torch.tensor(indexed), torch.tensor(self.label2idx[label])
# 2. 简单的文本分类模型
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        # 取最后一个时间步的输出
        out = self.fc(lstm_out[:, -1, :])
        return out
    
# # 2. 基于lstm文本分类模型
# class TextClassifier(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers=2, dropout=0.5):
#         super().__init__()
#         # Embedding层，支持预训练嵌入
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
#         # LSTM网络，增加层数和Dropout
#         self.lstm = nn.LSTM(
#             embedding_dim, 
#             hidden_dim, 
#             num_layers=num_layers, 
#             batch_first=True, 
#             dropout=dropout, 
#             bidirectional=True
#         )
        
#         # Attention层
#         self.attention = nn.Linear(hidden_dim * 2, 1)
        
#         # 全连接层，增加非线性激活和Dropout
#         self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
#         self.fc2 = nn.Linear(hidden_dim, num_classes)
        
#     def forward(self, x):
#         # 嵌入层
#         embedded = self.embedding(x)
        
#         # LSTM层
#         lstm_out, _ = self.lstm(embedded)
        
#         # # Attention机制
#         attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
#         context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
#         # 全连接层
#         out = self.fc1(context_vector)
#         out = self.relu(out)
#         out = self.dropout(out)
#         out = self.fc2(out)
        
#         return out

# 3. 训练函数
def train_model(model, train_loader, criterion, optimizer, device,test_loader,num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (texts, labels) in enumerate(train_loader):
            texts, labels = texts.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        print(f'Epoch: {epoch+1}, Average Loss: {total_loss/len(train_loader):.4f}')


# 4. 评估函数
def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return accuracy_score(all_labels, all_preds)

