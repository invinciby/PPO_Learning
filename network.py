import torch
from torch import nn 
import torch.nn.functional as F
import numpy as np

class FeedForwardNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FeedForwardNN, self).__init__()
        
        self.laryer1 = nn.Linear(in_dim, 128)
        self.laryer2 = nn.Linear(128, 128)
        self.laryer3 = nn.Linear(128, out_dim)
        
    def forward(self, obs):
        # 判断obs是否为numpy数组类型
        if isinstance(obs, np.ndarray):
            # print(f"----------{obs}----------")
            # 如果是，则将obs转换为torch张量，数据类型为float
            obs = torch.tensor(obs, dtype=torch.float)
            
        # 将obs输入到第一层神经网络，并使用ReLU激活函数
        activation1 = F.relu(self.laryer1(obs))
        # 将第一层的输出作为输入，输入到第二层神经网络，并使用ReLU激活函数
        activation2 = F.relu(self.laryer2(activation1))
        # 将第二层的输出作为输入，输入到第三层神经网络
        output = self.laryer3(activation2)
        
        # 返回第三层的输出
        return output