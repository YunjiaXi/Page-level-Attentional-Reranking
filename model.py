import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, static_bidirectional_rnn, LSTMCell, MultiRNNCell
import numpy as np
import json

class BaseModel(object):
    def __init__(self, args):
        self.args = args
        # self.max_hist_len = args.max_hist_len
        with open(args.stat_dir, 'r') as f:
            stat = json.load(f)

        self.feat_size = stat['ft_num']
        self.list_num = stat['list_num']
        self.list_len = stat['list_len']
        self.itm_spar_fnum = stat['itm_spar_fnum']
        self.itm_dens_fnum = stat['itm_dens_fnum']
        self.usr_fnum = stat['usr_fnum']
        self.hist_fnum = stat['hist_fnum']
        self.emb_dim = args.emb_dim
        self.max_hist_len = args.max_hist_len
        self.hidden_size = args.hidd_size
        self.max_grad_norm = args.grad_norm
        self.l2_norm = args.l2_norm
        self.lr = args.lr
        self.kp = args.keep_prob
        self.itm_ft_dim = self.itm_spar_fnum * self.emb_dim + self.itm_dens_fnum

        # reset graph
        tf.reset_default_graph()
        self.graph = tf.Graph()
        with self.graph.as_default():

            # input placeholders
            with tf.name_scope('inputs'):
                self.spar_ft_ph = tf.placeholder(tf.int32, [None, self.list_num, self.list_len, self.itm_spar_fnum], name='spar_ft')
                self.dens_ft_ph = tf.placeholder(tf.float32, [None, self.list_num, self.list_len, self.itm_dens_fnum], name='dens_ft')
                self.lb_ph = tf.placeholder(tf.float32, [None, self.list_num, self.list_len], name='lb')
                self.len_ph = tf.placeholder(tf.int32, [None, self.list_num], name='len')
                self.usr_ph = tf.placeholder(tf.int32, [None, self.usr_fnum], name='usr_ph')
                self.usr_hist = tf.placeholder(tf.int32, [None, self.max_hist_len, self.hist_fnum], name='usr_hist')
                self.is_train = tf.placeholder(tf.bool, [], name='is_train')

                self.keep_prob = tf.placeholder(tf.float32, [])

            # embedding
            with tf.name_scope('embedding'):
                self.emb_mtx = tf.get_variable('emb_mtx', [self.feat_size + 1, self.emb_dim],
                                               initializer=tf.truncated_normal_initializer)
                self.block_emb = tf.get_variable('block_emb', [self.list_num, self.emb_dim],
                                                 initializer=tf.truncated_normal_initializer)
                self.pos_emb = tf.get_variable('pos_emb', [self.list_len, self.emb_dim],
                                                 initializer=tf.truncated_normal_initializer)
                self.spar_emb = tf.gather(self.emb_mtx, self.spar_ft_ph)
                self.spar_emb = tf.reshape(self.spar_emb, [-1, self.list_num, self.list_len, self.emb_dim * self.itm_spar_fnum])
                self.emb = tf.concat([self.spar_emb, self.dens_ft_ph], axis=-1)
                self.usr_emb = tf.gather(self.emb_mtx, self.usr_ph)
                self.usr_emb = tf.reshape(self.usr_emb, [-1, 1, 1, self.usr_fnum * self.emb_dim])
                self.hist_emb = tf.gather(self.emb_mtx, self.usr_hist)
                self.hist_emb = tf.reshape(self.hist_emb, [-1, self.max_hist_len, self.hist_fnum * self.emb_dim])

            tile_user = tf.tile(self.usr_emb, [1, self.list_num, self.list_len, 1])
            tile_user = tf.reshape(tile_user, [-1, self.list_num, self.list_len, self.usr_fnum * self.emb_dim])
            self.page_user = tf.unstack(tile_user, axis=1)
            self.page = tf.unstack(self.emb, axis=1)
            self.page_lb = tf.unstack(self.lb_ph, axis=1)
            self.len_list = tf.unstack(self.len_ph, axis=1)
            self.max_len_list = [self.list_len] * self.list_num


    def build_fc_net(self, inp, max_len, scope='fc'):
        shape = inp.shape
        inp = tf.reshape(inp, [-1, shape[-1]])
        with tf.variable_scope(scope):
            bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1', training=self.is_train)
            fc1 = tf.layers.dense(bn1, 200, activation=tf.nn.relu, name='fc1')
            dp1 = tf.nn.dropout(fc1, self.keep_prob, name='dp1')
            fc2 = tf.layers.dense(dp1, 80, activation=tf.nn.relu, name='fc2')
            dp2 = tf.nn.dropout(fc2, self.keep_prob, name='dp2')
            fc3 = tf.layers.dense(dp2, 2, activation=None, name='fc3')
            score = tf.nn.softmax(fc3)
            score = tf.reshape(score[:, 0], [-1, max_len])
            # output
            # seq_mask = tf.sequence_mask(self.seq_length_ph, maxlen=self.max_time_len, dtype=tf.float32)
            # y_pred = seq_mask * score
        return score

    def final_pred_net(self, inp, layer=(200, 80), fin_act='sigmoid', scope='mlp'):
        shape = inp.shape
        inp = tf.reshape(inp, [-1, shape[-1]])
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            inp = tf.layers.batch_normalization(inputs=inp, name='mlp_bn', training=self.is_train)
            for i, hidden_num in enumerate(layer):
                fc = tf.layers.dense(inp, hidden_num, activation=tf.nn.relu, name='fc' + str(i))
                # bn = tf.layers.batch_normalization(inputs=fc, name='bn' + str(i), training=self.is_train)
                inp = tf.nn.dropout(fc, self.keep_prob, name='dp' + str(i))
            if fin_act == 'sigmoid':
                final = tf.layers.dense(inp, 1, activation=tf.nn.sigmoid, name='fc_final')
                score = tf.reshape(final, [-1, shape[-2]])
            elif fin_act == 'softmax':
                final = tf.layers.dense(inp, 1, activation=None, name='fc_final')
                score = tf.nn.softmax(tf.reshape(final, [-1, shape[-2]]))
            # seq_mask = tf.sequence_mask(self.seq_length_ph, maxlen=self.max_time_len, dtype=tf.float32)
            # y_pred = seq_mask * score
        return score

    def build_mlp_net(self, inp, layer=(200, 80), scope='mlp'):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            inp = tf.layers.batch_normalization(inputs=inp, name='mlp_bn', training=self.is_train)
            for i, hidden_num in enumerate(layer):
                fc = tf.layers.dense(inp, hidden_num, activation=tf.nn.relu, name='fc' + str(i))
                inp = tf.nn.dropout(fc, self.keep_prob, name='dp' + str(i))
        return inp

    def build_logloss(self, y_preds, lbs):
        # loss
        self.loss_list = []
        for pred, lb in zip(y_preds, lbs):
            self.loss_list.append(tf.losses.log_loss(lb, pred))
        self.loss = sum(self.loss_list)
        # self.loss = tf.losses.log_loss(lb, y_pred)
        self.opt()

    def opt(self):
        for v in tf.trainable_variables():
            if 'bias' not in v.name and 'emb' not in v.name:
                self.loss += self.l2_norm * tf.nn.l2_loss(v)
                # self.loss += self.l2_norm * tf.norm(v, ord=1)

        # self.lr = tf.train.exponential_decay(
        #     self.lr_start, self.global_step, self.lr_decay_step,
        #     self.lr_decay_rate, staircase=True, name="learning_rate")

        self.optimizer = tf.train.AdamOptimizer(self.lr)

        if self.max_grad_norm > 0:
            grads_and_vars = self.optimizer.compute_gradients(self.loss)
            for idx, (grad, var) in enumerate(grads_and_vars):
                if grad is not None:
                    grads_and_vars[idx] = (tf.clip_by_norm(grad, self.max_grad_norm), var)
            self.train_step = self.optimizer.apply_gradients(grads_and_vars)
        else:
            self.train_step = self.optimizer.minimize(self.loss)

    def multihead_attention(self,
                            queries,
                            keys,
                            num_units=None,
                            num_heads=2,
                            scope="multihead_attention",
                            dist_mat=None,
                            reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            if num_units is None:
                num_units = queries.get_shape().as_list()[-1]

            Q = tf.layers.dense(queries, num_units, activation=None)  # (N, T_q, C)
            K = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)
            V = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)

            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

            if dist_mat is not None:
                seq_len = queries.get_shape().as_list()[-2]
                batch_size = tf.shape(queries)[0]
                R = tf.tile(tf.expand_dims(dist_mat, 0), [num_heads, 1, 1])

                self.w = tf.constant(-0.1, dtype=tf.float32)
                self.v = tf.get_variable("v", [num_heads, 1, 1], initializer=tf.zeros_initializer)

                R_ = (1 + tf.exp(self.v)) / (1 + tf.exp(self.v - self.w * R))
                R_ = tf.tile(tf.expand_dims(R_, 1), [1, batch_size, 1, 1])
                R_ = tf.reshape(R_, [-1, seq_len, seq_len])
                outputs = tf.nn.softplus(outputs) * R_
                # outputs = tf.nn.sigmoid(outputs) * R_

            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

            key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
            key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)
            outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)
            coffi = outputs

            query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
            query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
            outputs *= query_masks  # broadcasting. (N, T_q, C)
            outputs = tf.nn.dropout(outputs, self.keep_prob)
            outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        return outputs, coffi

    def positionwise_feed_forward(self, inp, d_hid, d_inner_hid, scope='pos_ff'):
        with tf.variable_scope(scope):
            inp = tf.layers.batch_normalization(inputs=inp, name='bn1', training=self.is_train)
            l1 = tf.layers.dense(inp, d_inner_hid, activation=tf.nn.relu, name='fc1')
            dp = tf.nn.dropout(l1, self.keep_prob, name='dp')
            l2 = tf.layers.dense(dp, d_hid, activation=None, name='fc2')
            dp = l2 + inp
            output = tf.layers.batch_normalization(inputs=dp, name='bn2', training=self.is_train)
        return output

    def transformer(self, inp, scope='trans'):
        with tf.variable_scope(scope):
            att_out, att_coff = self.multihead_attention(inp, inp, self.args.d_model, self.args.n_head)
            ffn_out = self.positionwise_feed_forward(att_out, self.args.d_model, self.args.d_inner_hid)
        return ffn_out, att_coff

    def bilstm(self, inp, hidden_size, scope='bilstm', reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, name='cell_fw')
            lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, name='cell_bw')

            outputs, state_fw, state_bw = static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, inp, dtype='float32')
        return outputs, state_fw, state_bw

    def train(self, batch_data):
        with self.graph.as_default():
            loss, _ = self.sess.run([self.loss, self.train_step], feed_dict={
                self.usr_ph: batch_data['usr_ft'],
                self.spar_ft_ph: batch_data['spar_ft'],
                self.dens_ft_ph: batch_data['dens_ft'],
                self.lb_ph: batch_data['clk'],
                self.len_ph: batch_data['len'],
                self.usr_hist: batch_data['hist_ft'],
                self.keep_prob: self.kp,
                self.is_train: True,
            })
            return loss

    def eval(self, batch_data, no_print=True):
        with self.graph.as_default():
            pred, loss, lb = self.sess.run([self.y_pred, self.loss, self.page_lb], feed_dict={
                self.usr_ph: batch_data['usr_ft'],
                self.spar_ft_ph: batch_data['spar_ft'],
                self.dens_ft_ph: batch_data['dens_ft'],
                self.lb_ph: batch_data['clk'],
                self.len_ph: batch_data['len'],
                self.usr_hist: batch_data['hist_ft'],
                self.keep_prob: 1,
                self.is_train: False,
            })
            return pred, lb, loss

    def save(self, path):
        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.save(self.sess, save_path=path)
            print('Save model:', path)

    def load(self, path):
        with self.graph.as_default():
            ckpt = tf.train.get_checkpoint_state(path)
            if ckpt and ckpt.model_checkpoint_path:
                saver = tf.train.Saver()
                saver.restore(sess=self.sess, save_path=ckpt.model_checkpoint_path)
                print('Restore model:', ckpt.model_checkpoint_path)

    def set_sess(self, sess):
        self.sess = sess

    def get_att_vector2(self, inp, scope):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            shape = inp.shape
            u_context = tf.Variable(tf.truncated_normal([shape[-1]]), name='u_context')
            # [batch_size, max_time, hidden_size * 2]
            h = tf.layers.dense(inp, shape[-1], activation=tf.nn.tanh)
            # [batch_size, max_time, 1]
            alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(h, u_context), axis=2, keep_dims=True), dim=1)
            # [batch_szie, max_time, hidden_size*2]-->[batch_size, max_time]
            atten_output = tf.reduce_sum(tf.multiply(inp, alpha), axis=1)
            return atten_output


class GA(BaseModel):
    def __init__(self, args):
        super(GA, self).__init__(args)

        with self.graph.as_default():
            self.y_pred = []
            self.concat_page = []
            shape = tf.shape(self.page[0])
            tile_pos = tf.tile(tf.expand_dims(self.pos_emb, axis=0), [shape[0], 1, 1])
            for i, list in enumerate(self.page):
                tile_block = tf.tile(tf.reshape(self.block_emb[i], [1, 1, -1]), [shape[0], self.list_len, 1])
                self.concat_page.append(tf.concat([list, tile_pos, tile_block], axis=-1))
            cmp_inp = tf.concat(self.concat_page, axis=1)
            # cmp_res, self.cmp_coff = self.multihead_attention(cmp_inp, cmp_inp, self.args.d_model, self.args.n_head)
            cmp_res, self.cmp_coff = self.transformer(cmp_inp, scope='global_attention')
            cmp_list = tf.split(cmp_res, self.max_len_list, axis=1)
            for i, list in enumerate(cmp_list):
                self.y_pred.append(self.final_pred_net(list, scope='final_mlp'))
            self.build_logloss(self.y_pred, self.page_lb)


class PAR(BaseModel):
    def __init__(self, args):
        super(PAR, self).__init__(args)
        # mmoe = False
        # self.DA = cmp

        with self.graph.as_default():
            self.y_pred = []

            # Spatial-scaled Attention
            self.dist_mat = self.get_distant_matrix()
            cmp_inp = tf.concat(self.page, axis=1)
            cmp_res, self.cmp_coff = self.multihead_attention(cmp_inp, cmp_inp, self.args.d_model, self.args.n_head, dist_mat=self.dist_mat)
            cmp_list = tf.split(cmp_res, self.max_len_list, axis=1)

            # Hierarchical dual-side attention
            channel, item_att_list, self.chann_coff = self.hierarchical_co_attention(self.page, self.hist_emb)

            for i, list in enumerate(self.page):
                tile_channel = tf.tile(tf.expand_dims(channel, 1), [1, self.max_len_list[i], 1])
                user_list = tf.concat([list, self.page_user[i]], axis=-1)
                # deep net
                deep_res = self.build_mlp_net(user_list, layer=[256, 64], scope='deep_net')
                # MMoE
                inp = tf.concat([list, deep_res, tile_channel, item_att_list[i], cmp_list[i]], axis=-1)
                gate_inp = inp
                self.y_pred.append(self.MMoE(inp, gate_inp, item_att_list[i], self.len_list[i], i, scope='mmoe'))

            self.build_logloss(self.y_pred, self.page_lb)

    def MMoE(self, inp, gate_inp, item_att, inp_len, task, scope='mmoe'):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # sample-wise gate
            gate = self.build_mlp_net(gate_inp, layer=[self.args.expert_num], scope='gate_' + str(task)) #(B, L, E)

            gate = tf.expand_dims(tf.nn.softmax(gate, axis=-1), axis=-2) # (B, L, 1, expert_num)
            expert = [self.build_mlp_net(inp, layer=[200, 80], scope='expert_' + str(i)) for i in range(self.args.expert_num)]
            expert = tf.stack(expert, axis=-2) # (B, L, expert_num, E)
            fin_inp = tf.squeeze(tf.matmul(gate, expert),  axis=-2)
            out = self.final_pred_net(fin_inp, fin_act='sigmoid', layer=[80], scope='final_mlp_' + str(task))
        return out

    def hierarchical_co_attention(self, inp, hist, scope='hiera_co_att'):
        with tf.variable_scope(scope):
            channel_list = []
            item_att_list = []
            for i, list in enumerate(inp):
                item_att, hist_att = self.parallel_co_attention(list, hist, 'item_co_att_' + str(i))
                att_vec = tf.concat([item_att, hist_att], axis=-1)
                item_att_list.append(att_vec)
                # channel_list.append(self.get_att_vector(att_vec, 'item_trans_att_' + str(i)))
                channel_list.append(self.get_att_vector2(att_vec, 'item_trans_att'))
            channels = tf.stack(channel_list, axis=1)

            channel_att, chann_coff = self.transformer(channels, 'channel_att')
            page_vector = self.get_att_vector2(channel_att, 'chann_trans_att')
        return page_vector, item_att_list, chann_coff

    def parallel_co_attention(self, V, Q, scope='co_att'):
        with tf.variable_scope(scope):
            v_dim, q_dim = V.get_shape()[-1], Q.get_shape()[-1]
            v_seq_len, q_seq_len = V.get_shape()[-2], Q.get_shape()[-2]
            bat_size = tf.shape(Q)[0]
            w_b = tf.get_variable("w_b", [1, q_dim, v_dim], initializer=tf.truncated_normal_initializer)
            C1 = tf.matmul(Q, tf.matmul(tf.tile(w_b, [bat_size, 1, 1]), tf.transpose(V, perm=[0, 2, 1])))
            C = tf.tanh(C1)

            w_v = tf.get_variable('w_v', [v_dim, v_seq_len], initializer=tf.truncated_normal_initializer)
            w_q = tf.get_variable('w_q', [q_dim, v_seq_len], initializer=tf.truncated_normal_initializer)
            hv_1 = tf.reshape(tf.matmul(tf.reshape(V, [-1, v_dim]), w_v), [-1, v_seq_len, v_seq_len])
            hq_1 = tf.reshape(tf.matmul(tf.reshape(Q, [-1, q_dim]), w_q), [-1, q_seq_len, v_seq_len])
            hq_1 = tf.transpose(hq_1, perm=[0, 2, 1])   # modified
            h_v = tf.nn.tanh(hv_1 + tf.matmul(hq_1, C))
            h_q = tf.nn.tanh(hq_1 + tf.matmul(hv_1, tf.transpose(C, perm=[0, 2, 1])))
            # h_q = tf.nn.tanh(tf.matmul(hv_1, tf.transpose(C, perm=[0, 2, 1])))
            a_v = tf.nn.softmax(h_v, axis=-1)
            a_q = tf.nn.softmax(h_q, axis=-1)
            v = tf.matmul(a_v, V)
            q = tf.matmul(a_q, Q)
        return v, q

    def get_distant_matrix(self, mode='manhattan'):
        if mode == 'manhattan':
            itm_pos = {}
            for i in range(self.list_num):
                for j in range(self.list_len):
                    itm_pos[i * self.list_len + j] = (i, j)
            itm_num = len(itm_pos)
            dist_mat = np.zeros((itm_num, itm_num))
            for i in range(itm_num):
                for j in range(itm_num):
                    dist_mat[i][j] = abs(itm_pos[i][0] - itm_pos[j][0]) + abs(itm_pos[i][1] - itm_pos[j][1])
            return tf.convert_to_tensor(dist_mat, tf.float32, name='dist_mat')
        else:
            return None