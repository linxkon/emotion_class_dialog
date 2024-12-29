import torch
import torch.nn as nn
import os
from transformers import BertModel, BertTokenizer, BertConfig

class Config(object):
    def __init__(self):
        """
        配置类，包含模型和训练所需的各种参数。
        """
        self.model_name = "bert" # 模型名称
        self.data_path = './data/'  #数据集的根路径
        self.train_path = self.data_path + "train.txt"  # 训练集
        self.dev_path = self.data_path + "dev.txt"  # 验证集
        self.test_path = self.data_path + "test.txt"  # 测试集
        self.class_list = [x.strip() for x in open(self.data_path + "class.txt").readlines()]  # 类别名单
        # self.train_len = len([x.strip() for x in open(self.data_path + "train.txt").readlines()])  # 样本数量
        with open(self.data_path + "train.txt",encoding='utf-8') as file:
            self.train_len = sum(1 for line in file if line.strip())

        self.save_path = "./models" #模型训练结果保存路径
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.save_path += "/" + self.model_name + ".pt"  # 模型训练结果

        self.save_path2 = "./saved_dic" # 量化模型存储结果路径
        if not os.path.exists(self.save_path2):
            os.mkdir(self.save_path2)
        self.save_path2 += "/" + self.model_name + "_quantized.pt"  # 量化模型存储结果

        # 模型训练+预测的时候, 放开下一行代码, 在GPU上运行.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 训练设备，如果GPU可用，则为cuda，否则为cpu
        # 模型量化的时候, 放开下一行代码, 在CPU上运行.
        # self.device = 'cpu'

        self.num_classes = len(self.class_list)  # 类别数
        self.num_epochs = 10  # epoch数
        self.batch_size = 256  # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-4  # 学习率
        self.bert_path = 'E:/model_mine/bert-base-chinese'  # 预训练BERT模型的路径
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path) # BERT模型的分词器
        self.bert_config = BertConfig.from_pretrained(self.bert_path + '/bert_config.json') # BERT模型的配置
        self.hidden_size = 768 # BERT模型的隐藏层大小
        self.t_total = (self.train_len // self.batch_size) * self.num_epochs  # 总的训练步数
        self.warmup_steps = int(self.t_total*0.05)  # warm-up步数

class Model(nn.Module):
    def __init__(self,config):
        super(Model,self).__init__()

        self.bert = BertModel.from_pretrained(config.bert_path, config=config.bert_config)
        self.dropout = nn.Dropout(p=0.3)  # Dropout层，丢弃率为50%
        self.fc = nn.Linear(config.hidden_size,config.num_classes)

    def forward(self,x):
        context = x[0]
        mask = x[2]

        _,pooled=self.bert(context,attention_mask = mask,return_dict=False)
        pooled_output = self.dropout(pooled)
        out = self.fc(pooled_output)
        return out


# from utils import build_iterator,build_dataset
#
# config=Config()
# if __name__=='__main__':
#     train, dev, test=build_dataset(config)
#     train_iter = build_iterator(train, config)


