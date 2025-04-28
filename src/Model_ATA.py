import torch
import torch.nn as nn
import torch.nn.functional as F
from ConfigLoader import *

class ATA(nn.Module):
    def __init__(self):
        super(ATA, self).__init__()
        self.g_size = config_model['g_size']
        self.y_size = config_model['y_size']
        self.alpha = config_model['alpha']
        self.kl_lambda_anneal_rate = config_model['kl_lambda_anneal_rate']
        self.kl_lambda_start_step = config_model['kl_lambda_start_step']
        self.use_constant_kl_lambda = config_model['use_constant_kl_lambda']
        self.constant_kl_lambda = config_model['constant_kl_lambda']
        self.daily_att = config_model['daily_att']
        self.variant_type = config_model['variant_type']

        # Layers for temporal attention
        self.linear_v_i = nn.Linear(self.g_size, self.g_size, bias=False)
        self.linear_v_d = nn.Linear(self.g_size, self.g_size, bias=False)
        self.attention_weight = nn.Parameter(torch.Tensor(self.g_size, 1))
        nn.init.xavier_uniform_(self.attention_weight)

        # Layers for generating final predictions

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.linear_final = nn.Linear(self.g_size, self.y_size)
        self.softmax = nn.Softmax(dim=-1)



    def temporal_attention(self, g, g_T, mask_aux_trading_days, y):
        """
        Temporal attention mechanism.
        Args:
            g: Tensor of shape [batch_size, max_n_days, g_size].
            g_T: Tensor of shape [batch_size, g_size].
            mask_aux_trading_days: Mask for auxiliary trading days [batch_size, max_n_days].

        Returns:
            v_stared: Attention scores [batch_size, max_n_days].
            att_c: Context vector [batch_size, g_size].
        """
        proj_i = torch.tanh(self.linear_v_i(g))  # [batch_size, max_n_days, g_size]
        v_i = torch.matmul(proj_i, self.attention_weight).squeeze(-1)  # [batch_size, max_n_days]

        proj_d = torch.tanh(self.linear_v_d(g))  # [batch_size, max_n_days, g_size]
        g_T = g_T.unsqueeze(-1)  # [batch_size, g_size, 1]
        v_d = torch.matmul(proj_d, g_T).squeeze(-1)  # [batch_size, max_n_days]

        aux_score = v_i * v_d  # [batch_size, max_n_days]
        aux_score = torch.where(mask_aux_trading_days, aux_score, torch.full_like(aux_score, float('-inf')))
        v_stared = F.softmax(aux_score, dim=-1)  # [batch_size, max_n_days]
        v_stared = torch.nan_to_num(v_stared, nan=0.0)

        context = g.permute(0, 2, 1) if self.daily_att != 'y' else y.permute(0, 2, 1)  # Context based on g or y
        # print(context.shape)
        # print(v_stared.unsqueeze(1).permute(0, 2, 1).shape)
        att_c = torch.matmul(context, v_stared.unsqueeze(1).permute(0, 2, 1)).squeeze(2)  # [batch_size, g_size]

        return v_stared, att_c

    def generative_loss(self, y_ph, y, kl, v_stared, y_T, T_ph, kl_lambda):
        """
        Compute the generative loss.
        Args:
            y_ph: Ground truth labels [batch_size, max_n_days, y_size].
            y: Predicted labels [batch_size, max_n_days, y_size].
            kl: KL divergence [batch_size, max_n_days].
            v_stared: Attention scores [batch_size, max_n_days].
            g_T: Final latent state [batch_size, g_size].
            y_T: Final predicted label [batch_size, y_size].
            indexed_T: Indices of the final timestep [batch_size].
            kl_lambda: Weight for KL divergence.

        Returns:
            loss: Scalar loss value.
        """
        likelihood_aux = (y_ph * torch.log(y + 1e-7)).sum(dim=-1)  # [batch_size, max_n_days]
        obj_aux = likelihood_aux - kl_lambda * kl  # [batch_size, max_n_days]

        # print("y_ph shape:", y_ph.shape)  # 检查 y_ph 的形状
        # print("self.batch_size:", self.batch_size)
        # print("y_ph:", y_ph)  # 检查 y_ph 的形状




        # likelihood_T = (y_ph.gather(1, indexed_T.unsqueeze(-1).expand(-1, -1, self.y_size)) *
        #                 torch.log(y_T + 1e-7)).sum(dim=-1)  # [batch_size]
        likelihood_T = (y_ph[torch.arange(self.batch_size), T_ph - 1] *
                        torch.log(y_T + 1e-7)).sum(dim=-1)  # [batch_size]
        # kl_T = kl.gather(1, indexed_T.unsqueeze(-1)).squeeze(-1)  # [batch_size]
        kl_T = kl[torch.arange(self.batch_size), T_ph - 1]
        obj_T = likelihood_T - kl_lambda * kl_T  # [batch_size]

        v_aux = self.alpha * v_stared  # [batch_size, max_n_days]
        obj = obj_T + (obj_aux * v_aux).sum(dim=-1)  # [batch_size]
        loss = -obj.mean()  # Scalar loss value

        return loss

    def discriminative_loss(self, y_ph, y, v_stared, y_T, T_ph):
        """
        Compute the discriminative loss.
        Args:
            y_ph: Ground truth labels [batch_size, max_n_days, y_size].
            y: Predicted labels [batch_size, max_n_days, y_size].
            v_stared: Attention scores [batch_size, max_n_days].
            y_T: Final predicted label [batch_size, y_size].
            indexed_T: Indices of the final timestep [batch_size].

        Returns:
            loss: Scalar loss value.
        """
        likelihood_aux = (y_ph * torch.log(y + 1e-7)).sum(dim=-1)  # [batch_size, max_n_days]
        likelihood_T = (y_ph[torch.arange(self.batch_size), T_ph - 1] *
                        torch.log(y_T + 1e-7)).sum(dim=-1)  # [batch_size]

        v_aux = self.alpha * v_stared  # [batch_size, max_n_days]
        obj = likelihood_T + (likelihood_aux * v_aux).sum(dim=-1)  # [batch_size]
        loss = -obj.mean()  # Scalar loss value

        return loss

    def kl_lambda(self, global_step):
        """
        Compute the KL divergence weight.
        """
        if global_step < self.kl_lambda_start_step:
            return 0.0
        if self.use_constant_kl_lambda:
            return self.constant_kl_lambda
        return min(self.kl_lambda_anneal_rate * global_step, 1.0)

    def forward(self, g, g_T, y_ph, y, kl, mask_aux_trading_days, T_ph, global_step):
        """
        Forward pass for training.
        """
        self.batch_size = T_ph.shape[0]

        v_stared, att_c = self.temporal_attention(g, g_T, mask_aux_trading_days, y)
        kl_lambda = self.kl_lambda(global_step)

        conbin_att = torch.cat((att_c, g_T), dim=-1).to(self.device)
        self.linear_y_T = nn.Linear(conbin_att.shape[-1], self.y_size).to(self.device)

        y_T = self.softmax(self.linear_y_T(conbin_att))

        # print(indexed_T)
        if self.variant_type == 'discriminative':

            loss = self.discriminative_loss(y_ph, y, v_stared, y_T, T_ph)
        else:

            loss = self.generative_loss(y_ph, y, kl, v_stared, y_T, T_ph, kl_lambda)

        return loss



if __name__ == '__main__':
    # ata = ATA()
    import torch

    # 创建一个形状为 [32, 5, 2] 的随机张量
    tensor = torch.rand(32, 5, 2)

    # 打印张量形状和示例值
    print("Tensor shape:", tensor.shape)
    print("Tensor values:", tensor)
