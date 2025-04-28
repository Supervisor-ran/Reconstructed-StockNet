
import torch
import torch.nn as nn
import torch.nn.functional as F
from ConfigLoader import logger, ss_size, vocab_size, config_model


class MIE(nn.Module):
    def __init__(self, init_word_table):
        super(MIE, self).__init__()
        logger.info('INIT: #stock: {0}, #vocab+1: {1}'.format(ss_size, vocab_size))

        # Model config
        self.max_n_days = config_model['max_n_days']
        self.max_n_msgs = config_model['max_n_msgs']
        self.max_n_words = config_model['max_n_words']

        self.weight_init = config_model['weight_init']
        self.initializer = nn.init.xavier_uniform_ if self.weight_init == 'xavier-uniform' else nn.init.xavier_normal_

        self.word_embed_type = config_model['word_embed_type']

        self.y_size = config_model['y_size']
        self.word_embed_size = config_model['word_embed_size']

        self.mel_cell_type = config_model['mel_cell_type']
        self.variant_type = config_model['variant_type']

        self.mel_h_size = config_model['mel_h_size']
        self.msg_embed_size = config_model['mel_h_size']

        self.dropout_train_mel_in = config_model['dropout_mel_in']
        self.dropout_train_mel = config_model['dropout_mel']
        self.dropout_train_ce = config_model['dropout_ce']


        # Word Embedding
        self.word_embedding = nn.Embedding(vocab_size, self.word_embed_size)
        self.word_embedding.weight.data.copy_(init_word_table)

        # RNN Layers
        if self.mel_cell_type == 'ln-lstm':
            self.rnn = nn.LSTM(self.word_embed_size, self.mel_h_size, batch_first=True, bidirectional=True)
        elif self.mel_cell_type == 'gru':
            self.rnn = nn.GRU(self.word_embed_size, self.mel_h_size, batch_first=True, bidirectional=True)
        else:
            self.rnn = nn.RNN(self.word_embed_size, self.mel_h_size, batch_first=True, bidirectional=True)

        # Attention layers
        self.attention_proj = nn.Linear(self.msg_embed_size, self.msg_embed_size, bias=False)
        self.attention_weight = nn.Linear(self.msg_embed_size, 1, bias=False)

        # Dropout layers
        self.dropout_mel_in = nn.Dropout(self.dropout_train_mel_in)
        self.dropout_mel = nn.Dropout(self.dropout_train_mel)
        self.dropout_ce = nn.Dropout(self.dropout_train_ce)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def forward(self, word_input):
        """
        Forward pass for the model.
        Args:
            word_input: Tensor of word indices [batch_size, max_n_days, max_n_msgs, max_n_words].
            n_msgs_ph: Number of messages [batch_size, max_n_days].
        Returns:
            corpus_embed: Tensor of corpus embeddings [batch_size, max_n_days, corpus_embed_size].
        """
        batch_size, max_n_days, max_n_msgs, max_n_words = word_input.size()
        word_embed = self.word_embedding(word_input)  # [batch_size, max_n_days, max_n_msgs, max_n_words, embed_size]

        # print(word_embed.size())

        word_embed = self.dropout_mel_in(word_embed)
        word_embed = word_embed.view(-1, max_n_words, self.word_embed_size)  # Flatten for RNN input
        rnn_out, _ = self.rnn(word_embed)  # [batch_size * max_n_msgs, max_n_words, 2 * mel_h_size]
        # Separate forward and backward outputs
        rnn_out_forward = rnn_out[..., :self.mel_h_size]  # 前向 RNN 输出
        rnn_out_backward = rnn_out[..., self.mel_h_size:]  # 后向 RNN 输出

        # Average forward and backward outputs
        rnn_out = (rnn_out_forward + rnn_out_backward) / 2  # 取平均值

        # print("rnn_out", rnn_out.size())

        rnn_out = rnn_out.contiguous().view(batch_size, max_n_days, max_n_msgs, -1)  # Reshape back


        # print("rnn_out", rnn_out.size())
        self.attention_proj = nn.Linear(rnn_out.shape[-1], self.msg_embed_size, bias=False).to(self.device)
        rnn_out = rnn_out.to(self.device)
        attention_score = self.attention_proj(rnn_out).tanh()

        attention_weight = self.attention_weight(attention_score).squeeze(-1)
        attention_weight = F.softmax(attention_weight, dim=-1)  # Apply softmax over messages

        corpus_embed = torch.einsum('bijk,bij->bik', rnn_out, attention_weight)  # Weighted sum
        corpus_embed = self.dropout_ce(corpus_embed)

        return corpus_embed


# Initialize and train the model
# model = MIE()
# optimizer = optim.Adam(model.parameters(), lr=config_model['lr'])
