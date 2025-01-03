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
        '''
        nn.lstm 的输出是一个元组(output, (h_n, c_n))，其中
        output：每个时间步的隐藏状态，h_n：最终时间步的隐藏状态，c_n最终时间步的细胞状态
        '''
        # x: [batch_size, seq_len]
        embedded = self.embedding(x)
        # embedded: [batch_size, seq_len, embedding_dim]
        
        lstm_out, other = self.lstm(embedded)
        # lstm_out: [batch_size, seq_len, hidden_dim]
        # 取最后一个时间步的输出
        out = self.fc(lstm_out[:, -1, :])
        # breakpoint()
        return out
    
# 3. 训练函数
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch_idx, (texts, labels) in enumerate(train_loader):
        texts, labels = texts.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # if batch_idx % 100 == 0:
        #     print(f'Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
    return total_loss / len(train_loader)
    # print(f'Epoch: {epoch+1}, Average Loss: {total_loss/len(train_loader):.4f}')


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

