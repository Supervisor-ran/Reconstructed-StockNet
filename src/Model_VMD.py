import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import kl_divergence

from ConfigLoader import *

class VMD(nn.Module):
    def __init__(self):
        super(VMD, self).__init__()
        self.is_training_phase = None
        self.h_size = config_model['h_size']
        self.z_size = config_model['h_size']
        self.g_size = config_model['g_size']
        self.y_size = config_model['y_size']
        self.dropout_vmd_in = config_model['dropout_vmd_in']
        self.dropout_vmd = config_model['dropout_vmd']
        self.vmd_cell_type = config_model['vmd_cell_type']
        self.vmd_rec = config_model['vmd_rec']
        self.daily_att = config_model['daily_att']#
        self.variant_type = config_model['variant_type']
        uniform = True
        self.initializer = nn.init.xavier_uniform_ if uniform else nn.init.xavier_normal_
        self.bias_initializer = lambda bias: nn.init.constant_(bias, 0.0)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    def _linear(self, args, output_size, activation=None, use_bias=True, use_bn=False):
        # 如果输入不是列表或元组，封装为列表
        if not isinstance(args, (list, tuple)):
            args = [args]

        # 获取输入的总特征维度
        sizes = [a.size(-1) for a in args]  # 获取每个输入张量的最后一维
        total_arg_size = sum(sizes)  # 总输入维度

        # 将输入张量拼接
        x = args[0] if len(args) == 1 else torch.cat(args, dim=-1)

        # 初始化权重
        weight = torch.empty(total_arg_size, output_size, dtype=x.dtype, device=x.device)
        self.initializer(weight)  # 使用提供的初始化方法
        weight = nn.Parameter(weight)  # 转为模型参数

        # 线性变换
        res = torch.matmul(x, weight)

        # 如果使用偏置
        if use_bias:
            bias = torch.empty(output_size, dtype=x.dtype, device=x.device)
            self.bias_initializer(bias)  # 使用提供的初始化方法
            bias = nn.Parameter(bias)  # 转为模型参数
            res += bias

        # 如果需要使用 Batch Normalization
        if use_bn:
            bn = nn.BatchNorm1d(res.size(-1))  # 定义 BatchNorm
            if self.is_training_phase:  # 仅在训练时启用
                res = bn(res)

        # 应用激活函数
        if activation == 'tanh':
            res = torch.tanh(res)
        elif activation == 'sigmoid':
            res = torch.sigmoid(res)
        elif activation == 'relu':
            res = F.relu(res)
        elif activation == 'softmax':
            res = F.softmax(res, dim=-1)

        return res

    def _z(self, arg, is_prior):
        # 计算均值和标准差
        mean = self._linear(arg, self.z_size)  # 线性层计算均值
        stddev = self._linear(arg, self.z_size)  # 线性层计算标准差的对数
        stddev = torch.sqrt(torch.exp(stddev))  # 转换为标准差

        # 生成标准正态分布的噪声
        epsilon = torch.randn(self.batch_size, self.z_size, device=mean.device)  # 确保设备一致

        # 根据是否为先验生成 z
        z = mean if is_prior else mean + stddev * epsilon  # 使用广播进行乘法

        # 使用 PyTorch 的概率分布模块定义 Normal 分布
        pdf_z = Normal(loc=mean, scale=stddev)

        return z, pdf_z

    def _create_vmd_with_h_rec(self):
        # 输入数据和 Dropout
        x = F.dropout(self.x, p=self.dropout_vmd_in)  # Dropout
        x = x.permute(1, 0, 2)  # 转换维度: max_n_days * batch_size * x_size
        y_ = self.y_ph.permute(1, 0, 2)  # 转换维度: max_n_days * batch_size * y_size

        # 生成 mask
        self.mask_aux_trading_days = torch.arange(self.max_n_days, device=self.T_ph.device).expand(len(self.T_ph),
                                                                                                   self.max_n_days) < (
                                                 self.T_ph - 1).unsqueeze(1).bool()

        # 初始化变量
        h_s = torch.zeros(self.max_n_days, self.batch_size, self.h_size, device=x.device)
        z_prior = torch.zeros(self.max_n_days, self.batch_size, self.z_size, device=x.device)
        z_post = torch.zeros(self.max_n_days, self.batch_size, self.z_size, device=x.device)
        kl = torch.zeros(self.max_n_days, self.batch_size, self.z_size, device=x.device)

        # 循环遍历时间步
        for t in range(self.max_n_days):
            if t == 0:
                # 初始化 h_s 和 z
                h_s_t_1 = torch.tanh(torch.randn(self.batch_size, self.h_size, device=x.device))
                z_t_1 = torch.tanh(torch.randn(self.batch_size, self.z_size, device=x.device))
            else:
                h_s_t_1 = h_s[t - 1]
                z_t_1 = z_post[t - 1]

            # GRU 门和隐藏状态更新
            gate_args = torch.cat([x[t], h_s_t_1, z_t_1], dim=-1)
            r = self._linear(gate_args, self.h_size, activation="sigmoid")
            u = self._linear(gate_args, self.h_size, activation="sigmoid")
            h_args = torch.cat([x[t], r * h_s_t_1, z_t_1], dim=-1)
            h_tilde = self._linear(h_args, self.h_size, activation="tanh")
            h_s_t = (1 - u) * h_s_t_1 + u * h_tilde

            # 计算潜变量 z 的先验和后验
            h_z_prior_t = self._linear(torch.cat([x[t], h_s_t], dim=-1), self.z_size, activation="tanh")
            z_prior_t, z_prior_pdf = self._z(h_z_prior_t, is_prior=True)
            h_z_post_t = self._linear(torch.cat([x[t], h_s_t, y_[t]], dim=-1), self.z_size, activation="tanh")
            z_post_t, z_post_pdf = self._z(h_z_post_t, is_prior=False)

            # KL 散度
            kl_t = kl_divergence(z_post_pdf, z_prior_pdf)

            # 保存状态
            h_s[t] = h_s_t
            z_prior[t] = z_prior_t
            z_post[t] = z_post_t
            kl[t] = kl_t

        # 重塑张量
        h_s = h_s.permute(1, 0, 2)  # batch_size * max_n_days * h_size
        z_prior = z_prior.permute(1, 0, 2)  # batch_size * max_n_days * z_size
        z_post = z_post.permute(1, 0, 2)  # batch_size * max_n_days * z_size
        kl = kl.permute(1, 0, 2)  # batch_size * max_n_days * z_size

        self.kl = kl.sum(dim=-1)  # batch_size * max_n_days

        # 生成 g 和 y
        self.g = self._linear(torch.cat([x.permute(1, 0, 2), h_s, z_post], dim=-1), self.g_size, activation="tanh")
        self.y = self._linear(self.g, self.y_size, activation="softmax")

        # 提取最终时间步的特征
        sample_index = torch.arange(self.batch_size, device=x.device).unsqueeze(1)
        self.indexed_T = torch.cat([sample_index, (self.T_ph - 1).unsqueeze(1)], dim=1)
        # self.g_T = self.g[torch.arange(self.batch_size), self.T_ph - 1]
        # self.y_T = self.y[torch.arange(self.batch_size), self.T_ph - 1] if not self.daily_att else None

        def infer_func():
            g_T = self.g_T = self.g[torch.arange(self.batch_size), self.T_ph - 1]  # batch_size * g_size
            if not self.daily_att:
                y_T = self.self.y[torch.arange(self.batch_size), self.T_ph - 1]  # batch_size * y_size
                return g_T, y_T
            return g_T

        def gen_func():
            # Use prior for g
            z_prior_T = z_prior[torch.arange(self.batch_size), self.T_ph - 1]  # batch_size * z_size
            h_s_T = h_s[torch.arange(self.batch_size), self.T_ph - 1]
            x_T = x[torch.arange(self.batch_size), self.T_ph - 1]

            g_T = self._linear(torch.cat([x_T, h_s_T, z_prior_T], dim=-1), self.g_size)

            if not self.daily_att:
                y_T = F.softmax(self._linear(g_T, self.y_size), dim=-1)
                return g_T, y_T
            return g_T

        self.g_T = infer_func() if self.is_training_phase else gen_func()

    def _create_vmd_with_zh_rec(self):
        """
        Create a variational movement decoder.

        x: [batch_size, max_n_days, vmd_in_size]
        => vmd_h: [batch_size, max_n_days, vmd_h_size]
        => z: [batch_size, max_n_days, vmd_z_size]
        => y: [batch_size, max_n_days, 2]
        """
        # Dropout
        x = F.dropout(self.x, p=self.dropout_vmd_in)

        # Mask for auxiliary trading days
        # 假设 self.T_ph 是一个包含序列长度的张量，self.max_n_days 是最大序列长度
        # 创建序列掩码
        self.mask_aux_trading_days = torch.arange(self.max_n_days, device=self.T_ph.device).expand(len(self.T_ph),
                                                self.max_n_days) < (self.T_ph - 1).unsqueeze(1).bool()

        # Initialize RNN
        if self.vmd_cell_type == 'ln-lstm':
            rnn = nn.LSTM(self.x_size, self.h_size, batch_first=True, dropout=self.dropout_vmd).to(self.device)
        else:
            rnn = nn.GRU(self.x_size, self.h_size, batch_first=True, dropout=self.dropout_vmd).to(self.device)

        # rnn = nn.Dropout(p=self.dropout_vmd)(rnn)

        # Calculate vmd_h
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x.to("cpu"), self.T_ph.cpu(), batch_first=True, enforce_sorted=False
        ).to(self.device)
        packed_input = packed_input.to(self.device)
        h_s_packed, _ = rnn(packed_input)
        h_s, _ = nn.utils.rnn.pad_packed_sequence(h_s_packed, batch_first=True, total_length=x.shape[1])

        # 检查解包后的形状
        # print(f"h_s.shape: {h_s.shape}")
        assert h_s.shape[1] == 5, f"Expected seq_len=5, but got {h_s.shape[1]}"


        # Transpose for time-step-wise processing
        x = x.permute(1, 0, 2)  # [max_n_days, batch_size, x_size]
        h_s = h_s.permute(1, 0, 2)  # [max_n_days, batch_size, h_size]
        y_ = self.y_ph.permute(1, 0, 2)  # [max_n_days, batch_size, y_size]

        # Initialize tensors for z_prior, z_post, and kl
        z_prior = torch.zeros(self.max_n_days, self.batch_size, self.z_size, device=x.device)
        z_post = torch.zeros(self.max_n_days, self.batch_size, self.z_size, device=x.device)
        kl = torch.zeros(self.max_n_days, self.batch_size, self.z_size, device=x.device)

        # print(x.shape)
        # print(h_s.shape)
        # Loop through each time step
        for t in range(self.max_n_days):
            if t == 0:
                z_post_t_1 = torch.randn(self.batch_size, self.z_size, device=x.device)
            else:
                z_post_t_1 = z_post[t - 1]

            assert x.shape[0] == 5, f"{x.shape} ,{x}"
            assert h_s.shape[0] == 5, f"{h_s.shape} ,{h_s}"


            # Prior and posterior computations
            h_z_prior_t = self._linear(torch.cat([x[t,:,:], h_s[t,:,:], z_post_t_1], dim=-1), self.z_size, activation="tanh")
            z_prior_t, z_prior_pdf = self._z(h_z_prior_t, is_prior=True)

            h_z_post_t = self._linear(torch.cat([x[t,:,:], h_s[t,:,:], y_[t,:,:], z_post_t_1], dim=-1), self.z_size,
                                      activation="tanh")
            z_post_t, z_post_pdf = self._z(h_z_post_t, is_prior=False)

            kl_t = kl_divergence(z_post_pdf, z_prior_pdf)

            # Store results
            z_prior[t] = z_prior_t
            z_post[t] = z_post_t
            kl[t] = kl_t

        # Transpose back to batch-first format
        h_s = h_s.permute(1, 0, 2)  # [batch_size, max_n_days, h_size]
        z_prior = z_prior.permute(1, 0, 2)  # [batch_size, max_n_days, z_size]
        z_post = z_post.permute(1, 0, 2)  # [batch_size, max_n_days, z_size]
        kl = kl.permute(1, 0, 2)  # [batch_size, max_n_days, z_size]
        x = x.permute(1, 0, 2)

        self.kl = kl.sum(dim=-1)  # [batch_size, max_n_days]

        # Compute g and y
        self.g = self._linear(torch.cat([h_s, z_post], dim=-1), self.g_size, activation="tanh")
        self.y = self._linear(self.g, self.y_size, activation="softmax")

        # Extract the final g_T and y_T
        sample_index = torch.arange(self.batch_size, device=x.device).unsqueeze(1)
        self.indexed_T = torch.cat([sample_index, (self.T_ph - 1).unsqueeze(1)], dim=1)
        # self.g_T = self.g[torch.arange(self.batch_size), self.T_ph - 1]
        # self.y_T = self.y[torch.arange(self.batch_size), self.T_ph - 1] if not self.daily_att else None

        def infer_func():
            # g_T = self.g.gather(1, self.indexed_T.unsqueeze(-1).expand(-1, -1, self.g.size(-1)))  # batch_size * g_size
            g_T = self.g[torch.arange(self.batch_size), self.T_ph - 1]
            # print("g_T shape", g_T.shape)
            if not self.daily_att:
                y_T = self.y[torch.arange(self.batch_size), self.T_ph - 1]  # batch_size * y_size
                return g_T, y_T
            return g_T

        def gen_func():
            # Use prior for g
            # assert max(self.T_ph-1) <= 5, f"{self.T_ph}"
            z_prior_T = z_prior[torch.arange(self.batch_size), self.T_ph - 1]  # batch_size * z_size
            h_s_T = h_s[torch.arange(self.batch_size), self.T_ph - 1]

            x_T = x[torch.arange(self.batch_size), self.T_ph - 1]

            g_T = self._linear(torch.cat([x_T, h_s_T, z_prior_T], dim=-1), self.g_size)

            if not self.daily_att:
                y_T = F.softmax(self._linear(g_T, self.y_size), dim=-1)
                return g_T, y_T
            return g_T

        self.g_T = infer_func() if self.is_training_phase else gen_func()

    def _create_discriminative_vmd(self):
        """
        Create a discriminative movement decoder.

        x: [batch_size, max_n_days, vmd_in_size]
        => vmd_h: [batch_size, max_n_days, vmd_h_size]
        => z: [batch_size, max_n_days, vmd_z_size]
        => y: [batch_size, max_n_days, 2]
        """
        # Dropout on input
        x = F.dropout(self.x, p=self.dropout_vmd_in)
        x = x.permute(1, 0, 2)  # 转换维度: max_n_days * batch_size * x_size

        # Mask for auxiliary trading days
        self.mask_aux_trading_days = torch.arange(self.max_n_days, device=self.T_ph.device).expand(len(self.T_ph),
                                                                                                   self.max_n_days) < (
                                                 self.T_ph - 1).unsqueeze(1)

        # Initialize RNN
        if self.vmd_cell_type == 'ln-lstm':
            rnn = nn.LSTM(self.h_size, self.h_size, batch_first=True)
        else:
            rnn = nn.GRU(self.h_size, self.h_size, batch_first=True)

        # Forward RNN
        packed_input = nn.utils.rnn.pack_padded_sequence(x, self.T_ph.cpu(), batch_first=True, enforce_sorted=False)
        h_s_packed, _ = rnn(packed_input)
        h_s, _ = nn.utils.rnn.pad_packed_sequence(h_s_packed, batch_first=True)

        # Transpose for time-step-wise processing
        x = x.permute(1, 0, 2)  # [max_n_days, batch_size, x_size]
        h_s = h_s.permute(1, 0, 2)  # [max_n_days, batch_size, h_size]

        # Initialize z tensor
        z = torch.zeros(self.max_n_days, self.batch_size, self.z_size, device=x.device)

        # Loop through each time step
        for t in range(self.max_n_days):
            if t == 0:
                z_t_1 = torch.randn(self.batch_size, self.z_size, device=x.device)  # Random initialization
            else:
                z_t_1 = z[t - 1]

            # Compute h_z and z
            h_z_t = self._linear(torch.cat([x[t], h_s[t], z_t_1], dim=-1), self.z_size, activation="tanh")
            z_t = self._linear(h_z_t, self.z_size, activation="tanh")

            # Store z_t
            z[t] = z_t

        # Transpose back to batch-first format
        h_s = h_s.permute(1, 0, 2)  # [batch_size, max_n_days, h_size]
        z = z.permute(1, 0, 2)  # [batch_size, max_n_days, z_size]

        # Compute g and y
        self.g = self._linear(torch.cat([h_s, z], dim=-1), self.g_size, activation="tanh")
        self.y = self._linear(self.g, self.y_size, activation="softmax")

        # Extract g_T
        self.g_T = self.g[torch.arange(self.batch_size, device=x.device), self.T_ph - 1]



    def forward(self, x_, y_batch, T):

        self.x = x_
        self.y_ph = y_batch
        self.T_ph = T

        self.batch_size, self.max_n_days, self.x_size = self.x.shape

        if self.variant_type == 'discriminative':
            self._create_discriminative_vmd()
        else:
            if self.vmd_rec == 'h':
                self._create_vmd_with_h_rec()
            else:
                self._create_vmd_with_zh_rec()

        return self.g, self.g_T, self.y, self.kl, self.T_ph, self.mask_aux_trading_days
    #     """
    #     Args:
    #         x: Tensor of shape [batch_size, max_n_days, x_size].
    #         y: Tensor of shape [batch_size, max_n_days, y_size].
    #         T: Tensor of shape [batch_size], indicating sequence lengths.
    #
    #     Returns:
    #         g: Tensor of shape [batch_size, max_n_days, g_size].
    #         y_pred: Tensor of shape [batch_size, max_n_days, y_size].
    #         kl_divs: KL divergence for each day, shape [batch_size, max_n_days].
    #     """
    #     batch_size, max_n_days, x_size = x.shape
    #
    #     # RNN cell
    #     if self.vmd_cell_type == 'ln-lstm':
    #         self.rnn = nn.LSTM(x_size, self.h_size, batch_first=True)
    #     else:
    #         self.rnn = nn.GRU(x_size, self.h_size, batch_first=True)
    #
    #     self.dropout_rnn = nn.Dropout(p=self.dropout_vmd)
    #     self.dropout_in = nn.Dropout(p=self.dropout_vmd_in)
    #
    #     # Linear layers
    #     self.h_z_prior = nn.Linear(self.h_size + x_size, self.z_size)
    #     self.h_z_post = nn.Linear(self.h_size + x_size + self.y_size, self.z_size)
    #     self.z_prior_to_z = nn.Linear(self.z_size, self.z_size)
    #     self.z_post_to_z = nn.Linear(self.z_size, self.z_size)
    #     self.g_layer = nn.Linear(self.h_size + self.z_size, self.g_size)
    #     self.y_layer = nn.Linear(self.g_size, self.y_size)
    #
    #     # Apply dropout to input
    #     x = self.dropout_in(x)
    #
    #     # RNN encoding
    #     packed_x = nn.utils.rnn.pack_padded_sequence(x, T.cpu(), batch_first=True, enforce_sorted=False)
    #     h_s, _ = self.rnn(packed_x)
    #     h_s, _ = nn.utils.rnn.pad_packed_sequence(h_s, batch_first=True, total_length=max_n_days)
    #     h_s = self.dropout_rnn(h_s)
    #
    #     # Prepare for KL divergence calculation
    #     z_prior_list, z_post_list, kl_list = [], [], []
    #
    #     for t in range(max_n_days):
    #         x_t = x[:, t, :]
    #         h_s_t = h_s[:, t, :]
    #         y_t = y[:, t, :]
    #
    #         if t == 0:
    #             z_t_1 = torch.zeros(batch_size, self.z_size).to(x.device)
    #         else:
    #             z_t_1 = z_post_list[-1]
    #
    #         # Prior and posterior
    #         z_prior_t = torch.tanh(self.h_z_prior(torch.cat([x_t, h_s_t], dim=-1)))
    #         z_post_t = torch.tanh(self.h_z_post(torch.cat([x_t, h_s_t, y_t], dim=-1)))
    #
    #         z_prior_dist = Normal(z_prior_t, torch.ones_like(z_prior_t))
    #         z_post_dist = Normal(z_post_t, torch.ones_like(z_post_t))
    #
    #         z_prior_list.append(z_prior_t)
    #         z_post_list.append(z_post_t)
    #
    #         # KL divergence
    #         kl_t = kl_divergence(z_post_dist, z_prior_dist).sum(dim=-1)  # [batch_size]
    #         kl_list.append(kl_t)
    #
    #     # Stack results
    #     z_prior = torch.stack(z_prior_list, dim=1)  # [batch_size, max_n_days, z_size]
    #     z_post = torch.stack(z_post_list, dim=1)  # [batch_size, max_n_days, z_size]
    #     kl_divs = torch.stack(kl_list, dim=1)  # [batch_size, max_n_days]
    #
    #     # Compute g and y
    #     g = torch.tanh(self.g_layer(torch.cat([h_s, z_post], dim=-1)))  # [batch_size, max_n_days, g_size]
    #     y_pred = torch.softmax(self.y_layer(g), dim=-1)  # [batch_size, max_n_days, y_size]
    #
    #     return g, y_pred, kl_divs
