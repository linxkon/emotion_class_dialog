import torch
from transformers import BertTokenizer, BertModel
from torch import nn
from pathlib import Path
import pandas as pd

class BertClassifier(nn.Module):
    def __init__(self, bert_model, num_classes):
        super().__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(768, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[0][:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class EmotionPredictor:
    def __init__(self, model_path, bert_path, label_path):
        """
        初始化预测器
        model_path: 训练好的模型权重路径
        bert_path: BERT预训练模型路径
        label_path: 标签索引文件路径
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载标签映射
        label_df = pd.read_csv(label_path)
        self.idx2label = dict(zip(label_df['index'], label_df['label']))
        
        # 加载tokenizer和BERT模型
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        bert_model = BertModel.from_pretrained(bert_path)
        
        # 初始化分类器
        self.model = BertClassifier(bert_model, num_classes=len(self.idx2label))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # 设置最大长度
        self.max_len = 128
    
    def preprocess_text(self, text):
        """预处理输入文本"""
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].to(self.device),
            'attention_mask': encoding['attention_mask'].to(self.device)
        }
    
    def predict(self, text):
        """
        预测单条文本的情感
        返回: (预测的标签, 置信度)
        """
        # 预处理文本
        inputs = self.preprocess_text(text)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, dim=1)
            
        # 获取预测标签和置信度
        predicted_label = self.idx2label[prediction.item()]
        confidence_score = confidence.item()
        
        return predicted_label, confidence_score
    
    def predict_batch(self, texts):
        """
        批量预测多条文本的情感
        返回: [(预测的标签, 置信度), ...]
        """
        results = []
        for text in texts:
            result = self.predict(text)
            results.append(result)
        return results

# 使用示例
def main():
    # 设置路径
    model_path = Path('C_Bert/models') / 'bert_model.pth'
    bert_path = '/mnt/nvme1/linxkon/models/bert-base-chinese'
    label_path = Path('Data') / 'index.csv' #标签映射
    
    # 初始化预测器
    predictor = EmotionPredictor(
        model_path=model_path,
        bert_path=bert_path,
        label_path=label_path
    )
    
    # 单条文本预测示例
    text ='Q：春节啦！庆祝一下！热闹热闹！哈哈！    A：嘿嘿…'
    label, confidence = predictor.predict(text)
    print(f"文本: {text}")
    print(f"预测情感: {label}")
    print(f"置信度: {confidence:.4f}")
    
    # 批量预测示例
    texts = [
        "今天真是太开心了！",
        "这个消息太令人伤心了",
        "这部电影很好看，我很喜欢"
    ]
    results = predictor.predict_batch(texts)
    print("\n批量预测结果:")
    for text, (label, confidence) in zip(texts, results):
        print(f"文本: {text}")
        print(f"预测情感: {label}")
        print(f"置信度: {confidence:.4f}")
        print()

if __name__ == "__main__":
    main()