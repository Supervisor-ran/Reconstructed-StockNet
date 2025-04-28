import torch
from Model_MIE import MIE
from Model_VMD import VMD
from Model_ATA import ATA
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, word_table_init):
        super(Model, self).__init__()
        self.word_table_init = word_table_init
        self.model_name = "StockNet"
        self.global_step = torch.tensor(0, dtype=torch.int32)
        self.tf_graph_path = "../log/train"
        self.tf_checkpoint_file_path = "../checkpoint/3.pth"
        self.n_epochs = 10

        # 初始化子模型
        self.mie = MIE(self.word_table_init)
        self.vmd = VMD()
        self.ata = ATA()

    def forward(self, inputs, is_training_phase):
        # 检查输入数据类型
        assert isinstance(inputs, dict)
        assert isinstance(is_training_phase, bool)



        # MIE 模型处理
        corpu_emd = self.mie(inputs['word_batch'])
        mie_output = torch.cat((inputs['price_batch'], corpu_emd), dim=2)


        self.vmd.is_training_phase = is_training_phase
        # VMD 模型处理
        g, g_T, y_pred, kl, T_ph, mask_aux_trading_days = self.vmd(
            mie_output, inputs['y_batch'], inputs['T_batch']
        )



        # ATA 模型处理
        loss = self.ata(g, g_T, inputs['y_batch'], y_pred, kl, mask_aux_trading_days, T_ph, self.global_step)



        # 更新全局步数
        self.global_step += 1

        return inputs['y_batch'], y_pred, loss
