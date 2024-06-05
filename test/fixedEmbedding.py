import torch
import torch.nn as nn
import math
from datetime import datetime

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.requires_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() *
                    -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

# 将 datetime 转换为整数索引的函数
def datetime_to_index(dates, base_date=datetime(2020, 1, 1)):
    indices = [(d - base_date).days for d in dates]
    return torch.tensor(indices, dtype=torch.long)


# 示例日期时间列表
dates = [
    datetime(2020, 1, 1),
    datetime(2020, 1, 2),
    datetime(2020, 8, 3),
    datetime(2020, 12, 31)
]

# 将日期时间转换为整数索引
date_indices = datetime_to_index(dates)
print(date_indices)

# 定义嵌入层
c_in = 366  # 假设我们要嵌入一年的日期
d_model = 16  # 嵌入维度
embedding = FixedEmbedding(c_in, d_model)

# 获取嵌入向量
embedded_dates = embedding(date_indices)

print("嵌入向量：\n", embedded_dates)
