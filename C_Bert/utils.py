import torch
from tqdm import tqdm
import time
from datetime import timedelta # timedelta是时间间隔

# 数据加载器: 
# 加载文本文件、分词、添加特殊标记，生成包含词汇索引、标签、序列长度和填充掩码的数据集。
# 最终，函数返回训练集、验证集和测试集，以供模型使用
def build_dataset(config):
    """
    根据配置信息构建模型训练所需的数据集。

    参数：
    - config (object): 配置信息对象，包含有关数据集和模型的相关参数。

    返回：
    - train, dev, test (tuple): 包含三个元组，分别是训练集、验证集和测试集。
    """
    def load_dataset(path, pad_size=32):
        """
        加载并处理单个数据集文件。

        参数：
        - path (str): 数据集文件路径。
        - pad_size (int): 填充到的序列长度，默认为32。

        返回：
        - contents (list): 包含处理后的数据的列表。
        """
        contents = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:  # 如果为空行，则跳过
                    continue
                content, label = lin.split('\t')
                token = config.tokenizer.tokenize(content) # 分词
                token = ['[CLS]'] + token      # 添加特殊标记:分类
                #对句子长短进行处理
                seq_len = len(token)
                mask = []
                token_ids = config.tokenizer.convert_tokens_to_ids(token)  # 将词转换为索引
                if pad_size:
                    if len(token) < pad_size:  # 如果长度小于pad_size，则填充MASK
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += ([0] * (pad_size - len(token))) # 虚位词索引填充0
                    else:
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size

                contents.append((token_ids, int(label), seq_len, mask))
        return contents
    
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)

    return train, dev, test



# print(dev)

# 构建数据迭代器
class DatasetIterater(object):
    def __init__(self,batches,batchsize,device,model_name) -> None:
        """
        数据集迭代器的初始化函数。
        参数：
        - batches (list): 包含样本的列表。
        - batch_size (int): 每个批次的大小。
        - device (str): 数据加载到的设备（CPU或GPU）。
        - model_name (str): 使用的模型名称。
        """
        self.batches = batches # 所有数据
        self.batch_size = batchsize
        self.model_name = model_name
        self.n_batches = len(batches) // batchsize
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        """
        将数据转换为张量。

        参数：
        - datas (list): 包含数据的列表。

        返回：
        - tensor (Tensor): 包含数据的张量。
        """
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        if self.model_name == 'bert':
            mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
            return (x, seq_len, mask), y
        
        if self.model_name == 'textCNN':
            return (x, seq_len), y
        
    def __next__(self):
        """
        获取下一个批次的数据。

        返回：
        - data (tuple): 包含输入数据和标签的元组。
        """
        if self.residue and self.index == self.n_batches:  # 如果batch数量不是整数，则最后剩余的数据用于最后一个批次
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            batches = self._to_tensor(batches)
            self.index += 1
            return batches

        elif self.index < self.n_batches: # 如果batch数量是整数
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            batches = self._to_tensor(batches)
            self.index += 1
            return batches
        else:
            self.index = 0
            raise StopIteration

        # return self._to_tensor(batches)
    
    def __iter__(self): #
        return self
    
    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches
        
def build_iterator(dataset,config):
    iter = DatasetIterater(dataset, config.batch_size, config.device, config.model_name)
    return iter

def get_time_dif(start_time):
    """
    计算已使用的时间差。
    """
    # 获取当前时间
    end_time = time.time()
    # 计算时间差
    time_dif = end_time - start_time
    # 将时间差转换为整数秒，并返回时间差对象
    
    return timedelta(seconds=int(round(time_dif)))

# train, dev, test=build_dataset(config)
# #测试
# train_iter = build_iterator(train, config)
# print(next(train_iter))

# from bert_config import Model
# model = Model()
# model(config,train)

    
    