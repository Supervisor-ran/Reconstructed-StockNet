import logging
import torch
from DataPipe import DataPipe
from ConfigLoader import logger
import metrics as metrics
import stat_logger as stat_logger
from Integrate_Models import Model
from torch.utils.tensorboard import SummaryWriter


class Executor:

    def __init__(self, model, silence_step=200, skip_step=20):
        self.model = model
        self.silence_step = silence_step
        self.skip_step = skip_step
        self.pipe = DataPipe()

        # PyTorch 相关配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def unit_test_train(self):
        # 训练过程
        # word_table_init = self.pipe.init_word_table()
        # self.model.word_table_init = torch.tensor(word_table_init, dtype=torch.float32).to(self.device)

        logger.info('Word table init: done!')
        logger.info('Model: {0}, start a new session!'.format(self.model.model_name))

        n_iter = self.model.global_step.item()  # 获取当前迭代步数

        # 初始化训练数据
        train_batch_gen = self.pipe.batch_gen(phase='train')
        train_batch_dict = next(train_batch_gen)

        train_epoch_size = 0.0
        train_epoch_n_acc = 0.0
        train_batch_loss_list = []

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)  # 优化器

        while n_iter < 10:
            self.model.train()  # 进入训练模式

            # 构造输入数据
            inputs = {
                'batch_size': train_batch_dict['batch_size'],
                'stock_batch': torch.tensor(train_batch_dict['stock_batch'], dtype=torch.long).to(self.device),
                'T_batch': torch.tensor(train_batch_dict['T_batch'], dtype=torch.long).to(self.device),
                'n_words_batch': torch.tensor(train_batch_dict['n_words_batch'], dtype=torch.long).to(self.device),
                'n_msgs_batch': torch.tensor(train_batch_dict['n_msgs_batch'], dtype=torch.long).to(self.device),
                'y_batch': torch.tensor(train_batch_dict['y_batch'], dtype=torch.float32).to(self.device),
                'price_batch': torch.tensor(train_batch_dict['price_batch'], dtype=torch.float32).to(self.device),
                'mv_percent_batch': torch.tensor(train_batch_dict['mv_percent_batch'], dtype=torch.float32).to(self.device),
                'word_batch': torch.tensor(train_batch_dict['word_batch'], dtype=torch.long).to(self.device),
                'ss_index_batch': torch.tensor(train_batch_dict['ss_index_batch'], dtype=torch.long).to(self.device),
                'main_mv_percent_batch': torch.tensor(train_batch_dict['main_mv_percent_batch'], dtype=torch.float32).to(
                    self.device),
            }

            optimizer.zero_grad()  # 清空梯度

            # 前向传播
            train_batch_y, train_batch_y_, train_batch_loss = self.model(inputs, self.model.training)

            # 反向传播
            train_batch_loss.backward()
            optimizer.step()

            # 统计信息
            train_epoch_size += float(train_batch_dict['batch_size'])
            train_batch_loss_list.append(train_batch_loss.item())  # 记录损失
            train_batch_n_acc = metrics.n_accurate(y=train_batch_y, y_=train_batch_y_)  # 准确度计算
            train_epoch_n_acc += float(train_batch_n_acc)

            if n_iter % self.skip_step == 0:
                stat_logger.print_batch_stat(n_iter, train_batch_loss.item(), train_batch_n_acc, train_batch_dict['batch_size'])

            n_iter += 1

    def generation(self, phase):
        self.model.eval()  # 进入评估模式

        generation_gen = self.pipe.batch_gen_by_stocks(phase)

        gen_loss_list = []
        gen_size, gen_n_acc = 0.0, 0.0
        y_list, y_list_ = [], []

        for gen_batch_dict in generation_gen:
            inputs = {
                'batch_size': gen_batch_dict['batch_size'],
                'stock_batch': torch.tensor(gen_batch_dict['stock_batch'], dtype=torch.long).to(self.device),
                'T_batch': torch.tensor(gen_batch_dict['T_batch'], dtype=torch.long).to(self.device),
                'n_words_batch': torch.tensor(gen_batch_dict['n_words_batch'], dtype=torch.long).to(self.device),
                'n_msgs_batch': torch.tensor(gen_batch_dict['n_msgs_batch'], dtype=torch.long).to(self.device),
                'y_batch': torch.tensor(gen_batch_dict['y_batch'], dtype=torch.float32).to(self.device),
                'price_batch': torch.tensor(gen_batch_dict['price_batch'], dtype=torch.float32).to(self.device),
                'mv_percent_batch': torch.tensor(gen_batch_dict['mv_percent_batch'], dtype=torch.float32).to(self.device),
                'word_batch': torch.tensor(gen_batch_dict['word_batch'], dtype=torch.long).to(self.device),
                'ss_index_batch': torch.tensor(gen_batch_dict['ss_index_batch'], dtype=torch.long).to(self.device),
                'dropout_mel_in': 0.0,
                'dropout_mel': 0.0,
                'dropout_ce': 0.0,
                'dropout_vmd_in': 0.0,
                'dropout_vmd': 0.0,
            }

            # 前向传播
            gen_batch_y, gen_batch_y_, gen_batch_loss = self.model(inputs, self.model.training)

            y_list.append(gen_batch_y)
            y_list_.append(gen_batch_y_)
            gen_loss_list.append(gen_batch_loss.item())  # 损失值

            # 计算准确度
            gen_batch_n_acc = metrics.n_accurate(y=gen_batch_y, y_=gen_batch_y_)
            gen_n_acc += gen_batch_n_acc

            batch_size = float(gen_batch_dict['batch_size'])
            gen_size += batch_size

        results = metrics.eval_res(gen_n_acc, gen_size, gen_loss_list, y_list, y_list_,use_mcc=True)
        return results

    def train_and_dev(self):
        writer = SummaryWriter(self.model.tf_graph_path)

        # # 初始化所有参数和词嵌入表
        # word_table_init = self.pipe.init_word_table()
        # self.model.word_table_init = torch.tensor(word_table_init, dtype=torch.float32).to(self.device)



        # 载入检查点（如果有）
        checkpoint_path = self.model.tf_checkpoint_file_path
        # if os.path.exists(checkpoint_path):
        #     checkpoint = torch.load(checkpoint_path)
        #     self.model.load_state_dict(checkpoint['model_state_dict'])
        #     logger.info(f'Model restored from {checkpoint_path}')
        # else:
        #     logger.info('No checkpoint found, starting training from scratch.')

        # 开始训练过程
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)  # 优化器
        for epoch in range(self.model.n_epochs):

            logger.info(f'Epoch {epoch + 1}/{self.model.n_epochs} start')

            train_batch_loss_list = []
            epoch_size, epoch_n_acc = 0.0, 0.0

            step = 0

            train_batch_gen = self.pipe.batch_gen(phase='train')  # 训练集生成器
            for train_batch_dict in train_batch_gen:
                self.model.train()
                step += 1
                inputs = {
                    'batch_size': train_batch_dict['batch_size'],
                    'stock_batch': torch.tensor(train_batch_dict['stock_batch'], dtype=torch.long).to(self.device),
                    'T_batch': torch.tensor(train_batch_dict['T_batch'], dtype=torch.long).to(self.device),
                    'n_words_batch': torch.tensor(train_batch_dict['n_words_batch'], dtype=torch.long).to(self.device),
                    'n_msgs_batch': torch.tensor(train_batch_dict['n_msgs_batch'], dtype=torch.long).to(self.device),
                    'y_batch': torch.tensor(train_batch_dict['y_batch'], dtype=torch.float32).to(self.device),
                    'price_batch': torch.tensor(train_batch_dict['price_batch'], dtype=torch.float32).to(self.device),
                    'mv_percent_batch': torch.tensor(train_batch_dict['mv_percent_batch'], dtype=torch.float32).to(self.device),
                    'word_batch': torch.tensor(train_batch_dict['word_batch'], dtype=torch.long).to(self.device),
                    'ss_index_batch': torch.tensor(train_batch_dict['ss_index_batch'], dtype=torch.long).to(self.device)
                }

                optimizer.zero_grad()  # 清空梯度




                # 前向传播
                train_batch_y, train_batch_y_, train_batch_loss = self.model(inputs, self.model.training)



                # 反向传播
                train_batch_loss.backward()
                optimizer.step()

                # 统计信息
                epoch_size += float(train_batch_dict['batch_size'])
                train_batch_loss_list.append(train_batch_loss.item())
                train_batch_n_acc = metrics.n_accurate(y=train_batch_y, y_=train_batch_y_)
                epoch_n_acc += float(train_batch_n_acc)

                stat_logger.print_batch_stat(epoch, train_batch_loss.item(), train_batch_n_acc, train_batch_dict['batch_size'])

                # 保存模型
                if (epoch * step + 1) % self.silence_step == 0:
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'loss': train_batch_loss,
                    }, checkpoint_path)


                    res = self.generation(phase='dev')

                    stat_logger.print_eval_res(res,use_mcc=True)


            # 打印训练状态
            epoch_loss, epoch_acc = metrics.basic_train_stat(train_batch_loss_list, epoch_n_acc, epoch_size)
            stat_logger.print_epoch_stat(epoch_loss=epoch_loss, epoch_acc=epoch_acc)




        writer.close()

    def restore_and_test(self):
        checkpoint_path = self.model.tf_checkpoint_file_path
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f'Model restored from {checkpoint_path}')

        res = self.generation(phase='test')
        stat_logger.print_eval_res(res)


if __name__ == '__main__':
    silence_step = 1
    skip_step = 20

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # pipe = DataPipe()
    # word_table_init = pipe.init_word_table()
    # logger.info('Word table init: done!')
    # model = Model(word_table_init.to(device))
    #
    # exe = Executor(model, silence_step=silence_step, skip_step=skip_step)
    # exe.train_and_dev()
    # exe.restore_and_test()

