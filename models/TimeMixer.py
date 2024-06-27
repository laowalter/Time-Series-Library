import torch
import torch.nn as nn
from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize


class DFT_series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, top_k=5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, 5)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        return x_season, x_trend


class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    input [B,C,T]
    """

    def __init__(self, configs):
        super(MultiScaleSeasonMixing, self).__init__()
        '''
        周期性数据合成从高密度到低密度
        MultiScaleSeasonMixing(
          (down_sampling_layers): ModuleList(
            (0): Sequential(
              (0): Linear(in_features=96, out_features=48, bias=True)
              (1): GELU(approximate='none')
              (2): Linear(in_features=48, out_features=48, bias=True)
            )
            (1): Sequential(
              (0): Linear(in_features=48, out_features=24, bias=True)
              (1): GELU(approximate='none')
              (2): Linear(in_features=24, out_features=24, bias=True)
            )
            (2): Sequential(
              (0): Linear(in_features=24, out_features=12, bias=True)
              (1): GELU(approximate='none')
              (2): Linear(in_features=12, out_features=12, bias=True)
            )
          )
        )
        '''
        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),

                )
                for i in range(configs.down_sampling_layers)
            ]
        )

    def forward(self, season_list):

        # mixing high->low
        '''
        season_list[0].shape => [2688, 16, 96]
        season_list[1].shape => [2688, 16, 48]
        season_list[2].shape => [2688, 16, 24]
        season_list[3].shape => [2688, 16, 12]
        out_high = season_list[0]  # 初始化为最高分辨率
        out_low = season_list[1]  # 次高分辨率的季节性模式
        out_season_list = [out_high.permute(0, 2, 1)]  # [B,T,C] 把96的sesson直接放入
        通过逐层下采样处理，可以将时间序列数据从高分辨率逐步转换到低分辨率，从而在每个
        层级上捕捉不同尺度的季节性特征。这种处理方式允许模型在不同时间尺度上识别和学习
        这些周期性模式。
        下面的循环就是高分辨率96降采样次分辨率48，然后和原来的次分辨率48结合成新的48，
        然后将新的48作为高分辨率，在通过season_list[i+2]找到原来的24作为次高，重复前面。
        '''
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """

    def __init__(self, configs):
        super(MultiScaleTrendMixing, self).__init__()
        '''
        趋势和成有低密度到高密度
        MultiScaleTrendMixing(
          (up_sampling_layers): ModuleList(
            (0): Sequential(
              (0): Linear(in_features=12, out_features=24, bias=True)
              (1): GELU(approximate='none')
              (2): Linear(in_features=24, out_features=24, bias=True)
            )
            (1): Sequential(
              (0): Linear(in_features=24, out_features=48, bias=True)
              (1): GELU(approximate='none')
              (2): Linear(in_features=48, out_features=48, bias=True)
            )
            (2): Sequential(
              (0): Linear(in_features=48, out_features=96, bias=True)
              (1): GELU(approximate='none')
              (2): Linear(in_features=96, out_features=96, bias=True)
            )
          )
        )
        '''

        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                )
                for i in reversed(range(configs.down_sampling_layers))
            ])

    def forward(self, trend_list):

        # mixing low->high
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list


class PastDecomposableMixing(nn.Module):
    """对不同的数据进行趋势分解"""

    def __init__(self, configs):
        super(PastDecomposableMixing, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window

        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.channel_independence = configs.channel_independence

        if configs.decomp_method == 'moving_avg':
            self.decompsition = series_decomp(configs.moving_avg)
        elif configs.decomp_method == "dft_decomp":
            self.decompsition = DFT_series_decomp(configs.top_k)
        else:
            raise ValueError('decompsition is error')

        if configs.channel_independence == 0:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
                nn.GELU(),
                nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
            )

        # Mixing season
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)

        # Mxing trend
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)

        '''
        Sequential(
          (0): Linear(in_features=16, out_features=32, bias=True)
          (1): GELU(approximate='none')
          (2): Linear(in_features=32, out_features=16, bias=True)
        )
        '''
        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
            nn.GELU(),
            nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
        )

    def forward(self, x_list):
        # x_list[0].shape: [2688,96,16] 即B,T.C

        length_list = []
        for x in x_list:  # 取出全部的sequence长度(T)  length_list[96,48,24,12] High->Low
            _, T, _ = x.size()
            length_list.append(T)

        # Decompose to obtain the season and trend
        # 每一个list都是保存从时间频率从high到low的数据, 数据顺序是B,C,T
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decompsition(x)
            if self.channel_independence == 0:
                season = self.cross_layer(season)  # B,T,C
                trend = self.cross_layer(trend)  # B,T,C

            season_list.append(season.permute(0, 2, 1))  # B,C,T
            trend_list.append(trend.permute(0, 2, 1))  # B,C,T

        # bottom-up season mixing
        out_season_list = self.mixing_multi_scale_season(season_list)  # [B,T,C]
        # top-down trend mixing
        out_trend_list = self.mixing_multi_scale_trend(trend_list)  # [B, T, C]

        out_list = []
        for ori, out_season, out_trend, length in zip(x_list, out_season_list, out_trend_list,
                                                      length_list):
            out = out_season + out_trend
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])
        return out_list


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        self.channel_independence = configs.channel_independence
        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(configs)
                                         for _ in range(configs.e_layers)])

        self.preprocess = series_decomp(configs.moving_avg)
        self.enc_in = configs.enc_in

        if self.channel_independence == 1:
            # 如果通道独立，那么每一个通道（特征）都分别编码
            self.enc_embedding = DataEmbedding_wo_pos(1,
                                                      configs.d_model,
                                                      configs.embed,
                                                      configs.freq,
                                                      configs.dropout)
        else:
            # 通道不独立，所有通道（特征）一起作为输入
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in,
                                                      configs.d_model,
                                                      configs.embed,
                                                      configs.freq,
                                                      configs.dropout)

        self.layer = configs.e_layers
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            '''
            ModuleList(
              (0): Linear(in_features=96, out_features=96, bias=True)
              (1): Linear(in_features=48, out_features=96, bias=True)
              (2): Linear(in_features=24, out_features=96, bias=True)
              (3): Linear(in_features=12, out_features=96, bias=True)
            )
            '''
            self.predict_layers = torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.pred_len,
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )

            if self.channel_independence == 1:
                self.projection_layer = nn.Linear(configs.d_model, 1, bias=True)
            else:
                self.projection_layer = nn.Linear(configs.d_model, configs.c_out, bias=True)
                self.out_res_layers = torch.nn.ModuleList([
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ])

                self.regression_layers = torch.nn.ModuleList(
                    [
                        torch.nn.Linear(
                            configs.seq_len // (configs.down_sampling_window ** i),
                            configs.pred_len,
                        )
                        for i in range(configs.down_sampling_layers + 1)
                    ]
                )

            self.normalize_layers = torch.nn.ModuleList(
                [
                    Normalize(self.configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )

    def out_projection(self, dec_out, i, out_res):
        dec_out = self.projection_layer(dec_out)
        out_res = out_res.permute(0, 2, 1)
        out_res = self.out_res_layers[i](out_res)
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)
        dec_out = dec_out + out_res
        return dec_out

    def pre_enc(self, x_list):
        if self.channel_independence == 1:
            return (x_list, None)
        else:
            out1_list = []
            out2_list = []
            for x in x_list:
                x_1, x_2 = self.preprocess(x)
                out1_list.append(x_1)
                out2_list.append(x_2)
            return (out1_list, out2_list)

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        if self.configs.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.configs.down_sampling_window,
                                           return_indices=False)
        elif self.configs.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(in_channels=self.configs.enc_in,
                                  out_channels=self.configs.enc_in,
                                  kernel_size=3, padding=padding,
                                  stride=self.configs.down_sampling_window,
                                  padding_mode='circular',
                                  bias=False)
        else:
            return x_enc, x_mark_enc

        # B,T,C -> B,C,T
        # nn.Conv1d 和 MaxPool1d, AvgPool1d 层的输入通常要求形状为 [B, C, T]
        # 当数据形状为 [B, T, C] 时，直接应用一维卷积层或池化层会导致操作在错误的维度
        # 上进行。为了确保卷积和池化操作在时间维度上进行，需要将数据的形状转换为 [B, C, T]。
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        # 初始化下采样结果列表
        x_enc_sampling_list = []
        x_mark_sampling_list = []

        # x_enc_sampling_list 保存全部的采样数据，第一个数据就是原始数据的[B,T,C]形式
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))  # 在这里是第二次permute, 回到了BTC.
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.configs.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)  # B,C.T
            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))  # B,T,C
            x_enc_ori = x_enc_sampling

            if x_mark_enc is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :]

        x_enc = x_enc_sampling_list
        x_mark_enc = x_mark_sampling_list if x_mark_enc is not None else None

        return x_enc, x_mark_enc

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        处理多尺度输入。__multi_scale_process_inputs
        归一化输入数据。__normalize_layers
        嵌入处理后的输入数据。
        使用编码器块处理嵌入数据。
        使用解码器块生成预测。
        将预测结果反归一化，得到最终输出。
        """
        # 生成多尺度 x_enc、x_mark_enc是不同尺度tensor组成的list
        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)

        # 归一化输入数据。
        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)
                x_mark = x_mark.repeat(N, 1, 1)  # repeat N times of B position = N*B
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc, ):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        # embedding
        enc_out_list = []
        x_list = self.pre_enc(x_list)
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_list[0])), x_list[0], x_mark_list):
                # x.shape (B*N, T, 1), x_mark.shpae (B*N, T, 4)
                enc_out = self.enc_embedding(x, x_mark)
                enc_out_list.append(enc_out)
        else:
            for i, x in zip(range(len(x_list[0])), x_list[0]):
                enc_out = self.enc_embedding(x, None)  # [B,T,C]
                enc_out_list.append(enc_out)

        '''
        这是将趋势项和季节项合成的过程, 最终enc_out_list的元素shape和x_list一样。B,T,C
        '''
        # Past Decomposable Mixing as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        # Future Multipredictor Mixing as decoder for future
        dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list)  # all element [128,96,21]
        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)  # [128,96,21]
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out

    def future_multi_mixing(self, B, enc_out_list, x_list):
        dec_out_list = []
        if self.channel_independence == 1:
            x_list = x_list[0]
            for i, enc_out in zip(range(len(x_list)), enc_out_list):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                dec_out = self.projection_layer(dec_out)  # [2688, 96, 1]
                dec_out = dec_out.reshape(B, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous()  # [128, 96, 21]
                dec_out_list.append(dec_out)  # dec_out [128,96,1]

        else:
            for i, enc_out, out_res in zip(range(len(x_list[0])), enc_out_list, x_list[1]):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                dec_out = self.out_projection(dec_out, i, out_res)
                dec_out_list.append(dec_out)

        return dec_out_list

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out_list = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out_list
        else:
            raise ValueError('Only forecast tasks implemented yet')
