import torch
import numpy as np
from DataPipe import DataPipe
from ConfigLoader import *
from Model_MIE import MIE
from Model_VMD import VMD
from Model_ATA import ATA


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pipe = DataPipe()
word_table_init = pipe.init_word_table()
# print(word_table_init.shape)

train_batch_gen = pipe.batch_gen(phase='train')

# print(next(train_batch_gen))
train_batch_dict = next(train_batch_gen)

inputs = {
                #meta
                'batch_size': train_batch_dict['batch_size'],
                'stock_batch': torch.tensor(train_batch_dict['stock_batch'], dtype=torch.long).to(device),
                'T_batch': torch.tensor(train_batch_dict['T_batch'], dtype=torch.long).to(device),
                #target
                'y_batch': torch.tensor(train_batch_dict['y_batch'], dtype=torch.float32).to(device),
                'mv_percent_batch': torch.tensor(train_batch_dict['mv_percent_batch'], dtype=torch.float32).to(device),
                'main_mv_percent_batch': torch.tensor(train_batch_dict['main_mv_percent_batch'], dtype=torch.float32).to(device),
                #source
                'n_words_batch': torch.tensor(train_batch_dict['n_words_batch'], dtype=torch.long).to(device),
                'n_msgs_batch': torch.tensor(train_batch_dict['n_msgs_batch'], dtype=torch.long).to(device),
                'price_batch': torch.tensor(train_batch_dict['price_batch'], dtype=torch.float32).to(device),
                'word_batch': torch.tensor(train_batch_dict['word_batch'], dtype=torch.long).to(device),
                'ss_index_batch': torch.tensor(train_batch_dict['ss_index_batch'], dtype=torch.long).to(device)
            }

# print(train_batch_dict['word_batch'].shape,train_batch_dict['price_batch'].shape)
# print(train_batch_dict['y_batch'].shape,train_batch_dict['mv_percent_batch'].shape)
print(train_batch_dict['T_batch'], train_batch_dict['T_batch'].shape)


# global_step = 0
#
# mie = MIE(word_table_init)
# corpu_emd = mie.forward(inputs['word_batch'])
#
# mie_output = torch.cat((inputs['price_batch'],corpu_emd), dim=2 )
# # print(mie_output.shape)
# is_training_phase = True
# vmd = VMD(is_training_phase)
# # print(inputs['y_batch'].shape,inputs['mv_percent_batch'].shape)
#
# g, g_T, y_pred, kl, T_ph, mask_aux_trading_days = vmd.forward(mie_output,inputs['y_batch'],inputs['T_batch'])
# # print(g,y_pred.shape,kl_divg)
#
# ata = ATA()
# loss = ata.forward(g, g_T, inputs['y_batch'], y_pred, kl, mask_aux_trading_days, T_ph, global_step)
