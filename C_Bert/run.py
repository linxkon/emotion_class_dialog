import torch
import numpy as np
from train_eval import trainer, test ,find_error_data
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator


# 命令行参数解析
parser = argparse.ArgumentParser(description="Chinese Text Classification")
parser.add_argument("--model", type=str, default='bert', help="choose a model: bert")
args = parser.parse_args()

if __name__ == "__main__":
    if args.model == "bert":
        # 导入对应模型的配置和模型定义
        model_name = "bert"
        x = import_module("models." + model_name)
        config = x.Config()

        # 设置随机种子，保证实验的可重复性
        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True  # 保证每次结果一样

        # 构建训练、验证、测试数据集和数据迭代器
        train_data, dev_data, test_data = build_dataset(config)
        train_iter = build_iterator(train_data, config)
        dev_iter = build_iterator(dev_data, config)
        test_iter = build_iterator(test_data, config)
        # 创建模型实例并移至指定设备
        model = x.Model(config).to(config.device)

        # 训练模型
        trainer(config, model, train_iter, dev_iter,config.batch_size)
        # 在测试集上测试模型性能
        test(config, model, test_iter)

        #训练数据集查错
        # model.load_state_dict(torch.load(config.save_path, map_location=config.device)) #加载模型
        # test(config, model, train_iter)
        # find_error_data(config, model, train_iter)
