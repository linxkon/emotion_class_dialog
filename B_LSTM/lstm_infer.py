import torch
import jieba
import numpy as np
from lstm import *
from pathlib import Path 
class TextPredictor:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        # 加载保存的模型和配置
        checkpoint = torch.load(model_path, map_location=device)
        self.vocab = checkpoint['vocab']
        self.label2idx = checkpoint['label2idx']
        self.idx2label = {v: k for k, v in self.label2idx.items()}
        self.max_len = 50  # 与训练时保持一致
        self.device = device
        
        # 初始化模型
        self.model = TextClassifier(
            vocab_size=checkpoint['model_config']['vocab_size'],
            embedding_dim=checkpoint['model_config']['embedding_dim'],
            hidden_dim=checkpoint['model_config']['hidden_dim'],
            num_classes=checkpoint['model_config']['num_classes']
        ).to(device)
        
        # 加载模型权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
    
    def preprocess_text(self, text):
        # 提取回答部分
        answer = text
        # answer = text.split('A：')[-1] if 'A：' in text else text
        
        # 分词并转换为索引
        tokens = list(jieba.cut(answer))
        indexed = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        
        # 处理长度
        if len(indexed) > self.max_len:
            indexed = indexed[:self.max_len]
        else:
            indexed = indexed + [self.vocab['<PAD>']] * (self.max_len - len(indexed))
        
        return torch.tensor(indexed).unsqueeze(0)  # 添加批次维度
    
    def predict(self, text):
        """
        预测单个文本的情感标签
        """
        with torch.no_grad():
            # 预处理文本
            input_tensor = self.preprocess_text(text).to(self.device)
            
            # 模型预测
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            predicted_idx = torch.argmax(probs, dim=1).item()
            
            # 获取预测标签和概率
            predicted_label = self.idx2label[predicted_idx]
            confidence = probs[0][predicted_idx].item()
            
            return {
                'text': text,
                'predicted_label': predicted_label,
                'confidence': confidence,
                'probabilities': {
                    self.idx2label[i]: prob.item()
                    for i, prob in enumerate(probs[0])
                }
            }

# 使用示例
def main():
    # 示例文本
    test_text = 'Q：春节啦！庆祝一下！热闹热闹！哈哈！    A：嘿嘿…'
    
    path_model=Path('B_LSTM/model') / 'lstm_model.pth'
    # 初始化预测器
    predictor = TextPredictor(path_model)
    
    # 进行预测
    result = predictor.predict(test_text)
    
    # 打印结果
    print("\n预测结果:")
    print(f"文本: {result['text']}")
    print(f"预测标签: {result['predicted_label']}")
    print(f"置信度: {result['confidence']:.4f}")
    print("\n各标签概率:")
    for label, prob in result['probabilities'].items():
        print(f"{label}: {prob:.4f}")

if __name__ == '__main__':
    main()