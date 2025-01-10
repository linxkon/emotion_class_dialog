from transformers import BertTokenizer, BertModel, AdamW
import torch
from torch.utils.tensorboard import SummaryWriter  #使用tensorboard可视化
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from bert_model import *
from pathlib import Path

# 5. 主函数
def main():
    # 基础配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #训练设备
    path_data=Path('Data') / 'ECG_data.csv' #训练数据
    path_index=Path('Data') / 'index.csv' #标签映射
    path_bert_base_chinese='/home/home/linxkon/AI_project/model/bert-base-chinese' #预训练模型
    path_model_saved=Path('C_Bert/models') / 'bert_model.pth' #模型存储位置

    #超参数
    num_epochs = 10
    batch_size=32
    max_len = 128
    learning_rate = 2e-5
    num_classes = 5
    dropout = 0.5
    
    # 加载数据
    data_df = pd.read_csv(path_data)
    label_df = pd.read_csv(path_index)


    
    # 数据预处理
    
    ## 标签映射字典
    label_dic = {k: v for k, v in zip(label_df['label'].values,label_df['index'].values)}  ##  {'angry': 0, 'disgust': 1, 'happy': 2, 'like': 3, 'sad': 4}
    texts = data_df['sentence'].values
    labels = data_df['label'].map(label_dic).values
    
    # 划分训练集和验证集
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    # 加载tokenizer和预训练模型
    tokenizer = BertTokenizer.from_pretrained(path_bert_base_chinese)
    bert_model = BertModel.from_pretrained(path_bert_base_chinese)
    
    # 创建数据集
    train_dataset = EmotionDataset(train_texts, train_labels, tokenizer,max_len)
    val_dataset = EmotionDataset(val_texts, val_labels, tokenizer,max_len)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size)
    
    # 初始化模型
    model = BertClassifier(bert_model, num_classes,dropout)
    model = model.to(device)
    
    # 设置优化器
    optimizer = AdamW(model.parameters(), learning_rate)
    
    # 训练循环
    best_accuracy = 0
    writer = SummaryWriter()
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        
        # 训练阶段
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)

        print(f'Training loss: {train_loss:.4f}, accuracy: {train_acc:.4f}')
        writer.add_scalar('loss/Train', train_loss, epoch*batch_size)
        writer.add_scalar('ACC/Train', train_acc, epoch*batch_size)
        
        # 验证阶段
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(f'Validation loss: {val_loss:.4f}, accuracy: {val_acc:.4f}')
        # writer.add_scalar('loss/Validation', val_loss, epoch*batch_size)
        writer.add_scalar('ACC/Validation', val_acc, epoch*batch_size)
        
        # 保存最佳模型
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            model_best = model

        
        print('-' * 60)

    torch.save(model_best.state_dict(), path_model_saved)
    print('Best model saved!')

if __name__ == '__main__':
    main()