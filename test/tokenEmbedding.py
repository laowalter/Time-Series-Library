import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding,
                                   padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


# 假设我们有一些示例数据
batch_size = 1
seq_len = 3
c_in = 4  # 对应于 OHLC
d_model = 64  # 输出的嵌入维度

# 例如，OHLC 分别为 3000, 3500, 2880, 3200
kline_data = torch.tensor([[[3000.0, 3500.0, 2880.0, 3200.0],
                            [3089.0, 3300.0, 2800.0, 3000.0],
                            [3100.0, 3590.0, 2780.0, 3800.0],
                            ]])  # 形状: [1, 1, 4]


mu = kline_data.mean(dim=1, keepdim=True)
sigma = kline_data.std(dim=1, keepdim=True)
kline_data = (kline_data - mu) / sigma

# 初始化 TokenEmbedding 模块
token_embedding = TokenEmbedding(c_in, d_model)

# 前向传播
embedded_data = token_embedding(kline_data)

# 输出的嵌入数据
print(embedded_data)
