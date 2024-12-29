# 定义特殊的符号和类别映射
import torch
from importlib import import_module
import numpy as np

CLS =  "[CLS]"

def inference(model, config, input_text, pad_size=32):
    """
    模型推理函数，用于对输入文本进行情感分析的推理。
    参数：
    - model: 已加载的情感分析模型。
    - config: 模型配置信息。
    - input_text: 待分析的文本。
    - pad_size: 指定文本填充的长度。
    """
# 输入文本分词预处理
    content = config.tokenizer.tokenize(input_text)
    content = [CLS] + content
    seq_len = len(content)
    token_ids = config.tokenizer.convert_tokens_to_ids(content)
# 规范文本长度
    if seq_len < pad_size:
        mask = [1] * len(token_ids) + [0] * (pad_size - seq_len)
        token_ids += [0] * (pad_size - seq_len)
    else:
        mask = [1] * pad_size
        token_ids = token_ids[:pad_size]
        seq_len = pad_size
# 文本转张量
    x = torch.LongTensor(token_ids).to(config.device)
    seq_len = torch.LongTensor(seq_len).to(config.device)
    mask = torch.LongTensor(mask).to(config.device)

# 格式对齐
    x = x.unsqueeze(0)
    seq_len = seq_len.unsqueeze(0)
    mask = mask.unsqueeze(0)
    data = (x, seq_len, mask)

# 模型预测
    output = model(data)
# 获取模型预测结果id
    predict_result = torch.max(output.data, 1)[1]
    return predict_result

#预测主函数
if __name__=='__main__':
    # 加载模型配置
    model_name ='bert'
    x = import_module('models.'+ model_name)
    config = x.Config()
    # 设置随机种子
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    # 创建并初始化模型
    model = x.Model(config).to(config.device)
    model.load_state_dict(torch.load(config.save_path,map_location=config.device))
    input_text = '眼前感觉有蚊子在飞'
    # 进行模型推理
    res = inference(model, config, input_text)
    # 获取类别名称
    id2name={i: value for i, value in enumerate(config.class_list)}
    result = id2name[res.item()]
    print(input_text.strip(), '的预测结果：')
    print(result)

    # # 输入待分析文本
    # with open('data/bak/test111.txt', 'r', encoding='utf-8') as f:
    #     text = f.readlines()
    #     for input_text in text:
    #         # input_text = '日本地震：金吉列关注在日学子系列报道'
    #         #进行模型推理
    #         res = inference(model,config,input_text)
    #         # 获取类别名称
    #         result= id2name[res.item()]
    #         print(input_text.strip(),'的预测结果：')
    #         print(result)