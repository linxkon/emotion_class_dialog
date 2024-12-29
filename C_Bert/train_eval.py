import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from torch.optim.lr_scheduler import LambdaLR
from utils import get_time_dif
from transformers.optimization import AdamW
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter  #使用tensorboard可视化
import math
import logging

def loss_fn(outputs, labels):
    """
    定义损失函数，使用交叉熵损失。
    """
    return nn.CrossEntropyLoss()(outputs, labels)


def trainer(config, model, train_iter, dev_iter,batch_size):
    """
    模型训练函数。
    参数：
    - config: 配置信息对象。
    - model: 待训练的模型。
    - train_iter: 训练集的迭代器。
    - dev_iter: 验证集的迭代器。
    """
    # 记录开始训练的时间
    start_time = time.time()
    writer = SummaryWriter()

    # 参数优化器设置
    param_optimizer = list(model.named_parameters()) #获取模型所有层的参数
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": 0.01  #L2正则化,
         },
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0
         }]
    # 设置优化器
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)

    # 定义warm-up和余弦退火函数
    def lr_lambda(current_step: int):
        warmup_steps = config.warmup_steps
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            progress = float(current_step - warmup_steps) / float(max(1, config.t_total - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    # 设置学习率调度器
    scheduler = LambdaLR(optimizer, lr_lambda)
    current_lr = optimizer.param_groups[0]['lr']

    dev_best_loss = float("inf")
    # 将模型设置为训练模式
    model.train()
    total_batch_all=0
    # 遍历每个轮次
    for epoch in range(config.num_epochs):
        total_batch = 0
        print("Epoch [{}/{}]".format(epoch + 1, config.num_epochs))
        # 遍历每个批次
        for i, (trains, labels) in enumerate(tqdm(train_iter)):
            # 模型前向传播
            outputs = model(trains)
            # 梯度清零
            model.zero_grad()
            # 计算损失
            loss = loss_fn(outputs, labels)
            # 反向传播
            loss.backward()
            # 参数更新
            optimizer.step()
            scheduler.step()
            # 每100个batch输出在训练集和验证集上的效果
            if total_batch % 20 == 0 and total_batch != 0:
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                # 评估验证集效果
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                # 若验证集损失更低，保存模型参数
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = "*"
                else:
                    improve = ""
                # 计算时间差
                time_dif = get_time_dif(start_time)
                # 输出训练和验证集上的效果
                # msg = "Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}"
                writer.add_scalar("Train/Loss", loss.item(), total_batch_all*batch_size)
                writer.add_scalar("Train/Acc", train_acc, total_batch_all*batch_size)
                writer.add_scalar("Dev/Loss", dev_loss, total_batch_all*batch_size)
                writer.add_scalar("Dev/Acc", dev_acc, total_batch_all*batch_size)
                writer.add_scalar("para/Lr", current_lr, total_batch_all*batch_size)

                # print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                # 评估完成后将模型置于训练模式, 更新参数
                model.train()
            # 每个batch结束后累加计数
            total_batch += 1
            total_batch_all +=1


def evaluate(config, model, data_iter, test=False):
    """
    模型评估函数。
    参数：
    - config: 配置信息对象。
    - model: 待评估的模型。
    - data_iter: 数据迭代器。
    - test: 是否为测试集评估。
    """

    # 采用量化模型进行推理时需要关闭
    model.eval()

    loss_total = 0
    # 预测结果
    predict_all = np.array([], dtype=int)
    # label信息
    labels_all = np.array([], dtype=int)
    # 不进行梯度计算
    with torch.no_grad():
        # 遍历数据集
        for texts, labels in data_iter:
            # 将数据送入网络中
            outputs = model(texts)    #得到logits(分数)
            # 损失函数
            loss = F.cross_entropy(outputs, labels)
            # 损失和
            loss_total += loss
            # 获取label信息
            labels = labels.data.cpu().numpy()
            # 获取预测结果
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()  #拿到单个类别结果
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic) #将单个类别结果汇总
    # 计算准确率
    acc = metrics.accuracy_score(labels_all, predict_all)

    if test:
        # 如果是测试集评估，计算分类报告和混淆矩阵
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    else:
        # 如果是验证集评估，仅返回准确率和平均损失
        return acc, loss_total / len(data_iter)

def test(config, model, test_iter):
    """
    模型测试函数，用于在测试集上进行最终的模型测试。
    参数：
    - config: 配置信息对象。
    - model: 待测试的模型。
    - test_iter: 测试集的数据迭代器。
    """
    # 采用量化模型进行推理时需要关闭
    # model.eval()
    from importlib import import_module



    start_time = time.time()
    # 调用验证函数计算评估指标
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)

    # 打印测试结果信息:输出测试集上的损失、准确率、分类报告和混淆矩阵等信息
    msg = "Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}"
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def find_error_data(config, model, data_iter):
    model.eval()
    error_data = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_iter):
            texts, labels = batch
            # 假设 texts 是一个包含所有批次文本的元组
            outputs = model(texts)
            probabilities = F.softmax(outputs, dim=1)
            predicted = torch.max(outputs.data, 1)[1]

            for i in range(len(labels)):
                if predicted[i] != labels[i]:
                    try:
                        # 假设 texts 中的每个元素对应一个批次的所有文本
                        original_text = config.tokenizer.decode(texts[0][i])
                    except Exception as e:
                        print(f"Error decoding text: {e}")
                        original_text = "Unable to decode"

                    true_label = config.class_list[labels[i].item()]
                    pred_label = config.class_list[predicted[i].item()]
                    prob = probabilities[i][predicted[i]].item()

                    error_data.append({
                        'original_text': original_text,
                        'true_label': true_label,
                        'predicted_label': pred_label,
                        'probability': prob
                    })

    # 将错误数据写入文件，每行一个样本，每个样本四个特征
    with open('error_data.txt', 'w', encoding='utf-8') as f:
        for item in error_data:
            f.write(
                f"{item['original_text']}\t{item['true_label']}\t{item['predicted_label']}\t{item['probability']:.4f}\n")

    print(f"已将 {len(error_data)} 条预测错误的数据保存到 error_data.txt")