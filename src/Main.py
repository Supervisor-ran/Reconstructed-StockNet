#!/usr/local/bin/python

import torch
from DataPipe import DataPipe
from ConfigLoader import logger
from Integrate_Models import Model
from Excutor import Executor


if __name__ == '__main__':
    silence_step = 1
    skip_step = 20

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = DataPipe()
    word_table_init = pipe.init_word_table()
    logger.info('Word table init: done!')
    model = Model(word_table_init.to(device))

    exe = Executor(model, silence_step=silence_step, skip_step=skip_step)

    exe.train_and_dev()
    exe.restore_and_test()
