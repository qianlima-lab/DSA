# -*- coding: utf-8 -*-

import tensorflow as tf
import math
import numpy as np

def load_data(sentences, labels, lengths, batch_size):
    #
    while 1:
        batch_num = int(math.ceil(sentences.shape[0] / (batch_size+ 0.0)))
        # print (batch_num)
        for i in range(batch_num):
            if i != batch_num - 1:
                data_batch = sentences[i * batch_size: (i + 1) * batch_size]
                label_batch = labels[i * batch_size: (i + 1) * batch_size]
                length_batch = lengths[i * batch_size: (i + 1) * batch_size]
            else:
                data_batch = sentences[i * batch_size:]
                label_batch = labels[i * batch_size:]
                length_batch = lengths[i * batch_size:]
            yield [data_batch, label_batch, length_batch]

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

class model(object):
    def __init__(self, FLAGS, parameters):

        self.batch_size = None
        self.n_steps = parameters['n_steps']
        self.input_dim = 300  # data.shape[2]
        self.output_channels = 150
        self.hidden_dim = FLAGS.hidden_dim  # LSTM hidden size
        self.num_class = FLAGS.num_class

        self.learning_rate = FLAGS.learning_rate  # 1e-3
        self.is_train = tf.placeholder(tf.bool, [], "is_train")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")


        with tf.name_scope("input"):
            self.this_batch_size = tf.placeholder(tf.int32, [], "this_batch_size")
            self.drop_keep_prob = tf.placeholder(tf.float32, [], "drop_keep_prob")
            self.x_keep_prob = tf.placeholder(tf.float32, [], "x_keep_prob")
            self.x = tf.placeholder(tf.int32, [self.batch_size, self.n_steps], "input")
            with tf.variable_scope("word_embedding"):
                self.embedding = tf.get_variable("embedding", initializer=parameters['embedding'], dtype=tf.float32,
                                                 trainable=False)
            self.xemb = tf.nn.embedding_lookup(self.embedding, self.x)
            self.x_drop = tf.nn.dropout(self.xemb,
                                        self.x_keep_prob)
            self.sequence_length = tf.placeholder(tf.int32, [self.batch_size], 'sequence_length')
            self.y = tf.placeholder(tf.int32, [self.batch_size], 'labels')

        with tf.variable_scope("lstm_layer"):

            self.hlf_win_len = FLAGS.hlf_win_len

            fcell = tf.contrib.rnn.LSTMCell(self.hidden_dim, forget_bias=1.0)
            bcell = tf.contrib.rnn.LSTMCell(self.hidden_dim, forget_bias=1.0)

            outputs, state = tf.nn.bidirectional_dynamic_rnn(fcell, bcell,self.x_drop,
                                                             sequence_length=self.sequence_length,
                                                             dtype=tf.float32)
            outputs = tf.concat(outputs, axis=-1)  # b,T,2h
            x_input = outputs

            one_seq_mask = tf.sequence_mask(self.sequence_length, self.n_steps, dtype=tf.int32)  # b,T
            length_mask = tf.cast(
                tf.matmul(tf.expand_dims(one_seq_mask, axis=-1), tf.expand_dims(one_seq_mask, axis=1)),
                tf.float32)  # b,T,T

            hf_win_mxlens = [int(xi) for xi in FLAGS.hlf_win_len[1:][:-1].split(',')]
            n_scales = len(hf_win_mxlens)-1
            x_inputs_v = []
            qk_lst = []
            for si in range(n_scales):
                wx_k = tf.get_variable("wx_k_{}".format(si), [self.hidden_dim * 2, self.hidden_dim],
                                       dtype=tf.float32)
                wx_q = tf.get_variable("wx_q_{}".format(si), [self.hidden_dim * 2, self.hidden_dim],
                                       dtype=tf.float32)
                wx_v = tf.get_variable("wx_v_{}".format(si), [self.hidden_dim * 2, self.hidden_dim ],
                                       dtype=tf.float32)
                x_inputi_k = tf.reshape(tf.matmul(tf.reshape(x_input, [-1, self.hidden_dim * 2]), wx_k),
                                     [-1, self.n_steps, self.hidden_dim])
                x_inputi_q = tf.reshape(tf.matmul(tf.reshape(x_input, [-1, self.hidden_dim * 2]), wx_q),
                                        [-1, self.n_steps, self.hidden_dim])
                x_inputi_v = tf.reshape(tf.matmul(tf.reshape(x_input, [-1, self.hidden_dim * 2]), wx_v),
                                        [-1, self.n_steps, self.hidden_dim])
                x_inputs_v.append(x_inputi_v)
                qk = self.q_matmul_k(x_inputi_q, x_inputi_k, length_mask)  # b,T,T
                qk_lst.append(qk)


            hs = []
            hf_win_sizes = []

            tf.get_variable("fgc_w0", [self.hidden_dim, self.hidden_dim], dtype=tf.float32)

            for hi, hf_win_mxlen in enumerate(hf_win_mxlens):
                if hi == 0:
                    continue
                elif hf_win_mxlen == -1: # end
                    hf_win_mxlength_sub = self.sequence_length-hf_win_mxlens[hi-1]# b
                    hf_win_mxlength = tf.where(hf_win_mxlength_sub>=0, hf_win_mxlength_sub, tf.zeros_like(self.sequence_length))
                else:
                    hf_win_mxlength = tf.ones_like(self.sequence_length) * (hf_win_mxlen-hf_win_mxlens[hi-1]) # self.sequence_length

                hf_win_mxlength = tf.cast(tf.tile(tf.expand_dims(hf_win_mxlength, axis=-1), [1, self.n_steps]), tf.float32)

                wx00 = tf.get_variable("wx_sigmoid{}".format(hi), [self.hidden_dim*2, 1], dtype=tf.float32)

                hf_win = tf.reshape(tf.nn.sigmoid(tf.matmul(tf.reshape(x_input, [-1, self.hidden_dim*2]), wx00)),
                             [-1, self.n_steps]) * hf_win_mxlength + hf_win_mxlens[hi-1]
                # b,T
                hf_win_sizes.append(hf_win)

            # b,T
            self.hf_win_length = tf.stack(hf_win_sizes,axis=2) # b,t,s
            self.win_relations = []
            for hi, hf_wini in enumerate(hf_win_sizes):
                qk_res = self.win_msk_i(qk_lst[hi], self.n_steps, length_mask, hf_wini)
                self.win_relations.append(qk_res)
                h = x_inputs_v[hi]
                # for i_layer in range(self.n_layer):
                h = self.qkv(h, qk_res, hf_wini)
                hs.append(h)  # nr,(b,h)

            win_output = self.h1_h2_dis(hs) # B,T,4H->B,4H
            win_output = tf.nn.dropout(win_output, self.drop_keep_prob)

        with tf.name_scope('fc'):
            out_dim = win_output.get_shape().as_list()[-1]
            self.Wfc = weight_variable([out_dim, self.output_channels])
            self.Bfc = bias_variable([self.output_channels])

            self.fc = tf.add(tf.matmul(win_output, self.Wfc), self.Bfc)
            self.fc = tf.nn.sigmoid(self.fc)

        with tf.name_scope('softmax'):
            self.Wfc2 = weight_variable([self.output_channels, self.num_class])
            self.Bfc2 = bias_variable([self.num_class])
            self.prediction = tf.matmul(self.fc, self.Wfc2) + self.Bfc2

            self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.prediction)
            self.loss = tf.reduce_mean(self.cross_entropy)

            opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, self.params)
            clipped_gradients, norm = tf.clip_by_global_norm(gradients, 5)
            self.train_step = opt.apply_gradients(
                zip(clipped_gradients, self.params), global_step=self.epoch_step)

        with tf.name_scope('accuracy'):
            self.correct_predict = tf.cast(tf.equal(tf.cast(tf.argmax(self.prediction, 1), tf.int32), self.y),
                                           tf.float32)
            self.acc = tf.reduce_mean(self.correct_predict, name="accuracy")

        self.saver = tf.train.Saver(tf.global_variables())

    def q_matmul_k(self, input_q, input_k, mask):

        qk = tf.matmul(input_q, tf.transpose(input_k, [0, 2, 1]))  # b*T*T
        return qk * mask

    def h1_h2_dis(self,hs):# n_win_size*[b,t,h]
        hs_stk = tf.stack(hs, axis=0)  # n,b,t,h
        hs_sum = tf.reduce_sum(hs_stk, axis=0, keep_dims=True)  # 1,b,t,h
        h_others = hs_sum - hs_stk  # n,b,t,h

        h1h2 = tf.reduce_sum(hs_stk * h_others, axis=3)  # n,b,t,h -> n,b,t
        self.dis = 1 - tf.nn.sigmoid(-tf.norm(hs_stk - h_others, 1, axis=-1))*2  # n,b,t

        self.tah = tf.nn.tanh(h1h2)
        self.g = self.tah * self.dis  # n,b,t
        c = tf.expand_dims(self.g, axis=-1) * (hs_stk)  # n,b,t,1 * 1,b,t,h
        h_mxs = tf.unstack(tf.reduce_max(hs_stk, axis=2),axis=0) # n,b,h
        c_mxs = tf.unstack(tf.reduce_max(c, axis=2),axis=0)  # n,b,h
        return tf.concat(h_mxs+c_mxs, axis=-1)  # b,2nh


    def qkv(self, input, relation, win):
        """
        :param input: b,t,h
        :param relation: b,t,t
        :return: h = b,t,h
        """
        win = tf.expand_dims(win,axis=-1) #b,t,1
        win = tf.cast(tf.where(win >= 1, win, tf.ones_like(win)), dtype=tf.float32)
        input_dim = input.get_shape().as_list()[-1]
        tf.get_variable_scope().reuse_variables()
        w1 = tf.get_variable("fgc_w0", [self.hidden_dim, self.hidden_dim], dtype=tf.float32)
        wx_ = tf.matmul(tf.reshape(tf.matmul(relation, input), [-1, input_dim]), w1)  # bT,h
        h = tf.reshape(wx_, [-1, self.n_steps, self.hidden_dim]) / win  # b,T,h
        return h

    def win2idx(self,hf_win):
        """
        :param n_windows: b,t
        :return: initial window idx acc to window size. [-2,-1,0,1,2]
        """
        hlf_win = tf.cast(hf_win, tf.float32)
        t = tf.cast(tf.tile(tf.expand_dims(tf.range(start=0, limit=self.n_steps, delta=1), 0), [self.this_batch_size, 1]), tf.float32)
        idx_start = tf.where(t - hlf_win >= 0, t - hlf_win, tf.zeros_like(t))
        idx_end = tf.where(t + hlf_win + 1 <= self.n_steps, t + hlf_win + 1,
                           tf.ones_like(t) * tf.constant(self.n_steps,dtype=tf.float32))

        idx_lst = tf.expand_dims(tf.range(start=0, limit=self.n_steps, delta=1, dtype=tf.float32), 0)
        mask = tf.tile(tf.expand_dims(tf.tile(idx_lst, [self.this_batch_size, 1]), axis=-1), [1, 1, self.n_steps])  # b,t,t
        masks = tf.unstack(mask, axis=2)
        idx_starts = tf.unstack(idx_start, axis=-1)
        idx_ends = tf.unstack(idx_end, axis=-1)
        mask_lst = []
        for (maskii, si, ei) in zip(masks, idx_starts, idx_ends):
            maski = tf.greater_equal(maskii, tf.tile(tf.expand_dims(si, axis=-1), [1, self.n_steps]))
            maskj = tf.less(maskii, tf.tile(tf.expand_dims(ei, axis=-1), [1, self.n_steps]))
            mask_lst.append(tf.cast(maski, tf.int32) * tf.cast(maskj, tf.int32))
        return tf.stack(mask_lst, axis=1) # b,T,T

    def win_msk_i(self, qk, max_seq_len, length_mask, hf_win):
        # input is a batch word embedding: b, T, emb
        # relation: b,T,T
        one_seq_mask = tf.sequence_mask(self.sequence_length, max_seq_len, dtype=tf.int32)  # b,T
        hf_win1 = tf.ceil(hf_win)
        hf_win2 = tf.floor(hf_win)
        win_mask1 = self.win2idx(hf_win1)
        win_mask2 = self.win2idx(hf_win2)
        seq_mask = tf.tile(tf.expand_dims(one_seq_mask, -1), [1, 1, max_seq_len])
        seq_mask_tran = tf.transpose(seq_mask, [0, 2, 1])  # b*T*T
        mask01_1 = win_mask1 * seq_mask_tran + (1 - seq_mask)  # b,T,T
        mask01_2 = win_mask2 * seq_mask_tran + (1 - seq_mask)  # b,T,T

        # assert np.max(mask.eval()) <= 2
        map_ids = [-np.inf, 0, 0]
        mask1 = tf.gather(map_ids, mask01_1)
        mask2 = tf.gather(map_ids, mask01_2)
        weights1 = tf.nn.softmax(qk + mask1)  # b,T,T
        weights2 = tf.nn.softmax(qk + mask2)
        weights = (weights2*tf.expand_dims(hf_win1-hf_win,axis=-1)+
                   weights1*tf.expand_dims(hf_win-hf_win2,axis=-1)) * length_mask  # b,T,T

        return weights