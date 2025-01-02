import torch
from torch import nn
from torch import optim

# 创建神经网络类
class Model(nn.Module):
    # 初始化参数
    def __init__(self):
        # 调用父类方法
        super(Model, self).__init__()
        # 创建网络层
        self.linear1 = nn.Linear(2, 2)
        self.linear2 = nn.Linear(2, 2)
        # 初始化神经网络参数
        self.linear1.weight.data = torch.tensor([[0.15, 0.20], [0.25, 0.30]])
        self.linear2.weight.data = torch.tensor([[0.40, 0.45], [0.50, 0.55]])
        self.linear1.bias.data = torch.tensor([0.35, 0.35])
        self.linear2.bias.data = torch.tensor([0.60, 0.60])
    # 前向传播方法
    def forward(self, x):
        # 数据经过第一层隐藏层
        print('i:',x)
        x = self.linear1(x)
        print('h:',x)
        # 计算第一层激活值
        x = torch.sigmoid(x)
        print('h_s:',x)
        # 数据经过第二层隐藏层
        x = self.linear2(x)
        print('o:',x)
        # 计算第二层激活值
        x = torch.sigmoid(x)
        print('o_s:',x)

        return x

if __name__ == '__main__':
    # 定义网络输入值和目标值
    inputs = torch.tensor([[0.05, 0.10]])
    target = torch.tensor([[0.01, 0.99]])
    # 实例化神经网络对象
    model = Model()
    output = model(inputs)
    print("output-->", output)
    # 计算误差
    loss = torch.sum((output - target) ** 2) / 2
    print("loss-->", loss)
    # 优化方法和反向传播算法
    optimizer = optim.SGD(model.parameters(), lr=0.5)
    optimizer.zero_grad()
    loss.backward()
    print("w1,w2,w3,w4-->", model.linear1.weight.grad.data)
    print("w5,w6,w7,w8-->", model.linear2.weight.grad.data)
    optimizer.step()
    # 打印神经网络参数
    print(model.state_dict())