import os
import tensorflow as tf
import random
import pickle as pkl
from utils import *
from model import *
from dataset import Dataset
from click_model import FCM
import argparse
import datetime


def train(args):
    tf.reset_default_graph()

    # gpu settings
    gpu_options = tf.GPUOptions(allow_growth=True)
    hist = True
    if args.algo == 'PAR':
        model = PAR(args)
    else:
        print('No Such Model', args.algo)
        exit(0)


    with model.graph.as_default() as g:
        sess = tf.Session(graph=g, config=tf.ConfigProto(gpu_options=gpu_options))
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        model.set_sess(sess)

    test_set = Dataset(args, args.test_data_dir, is_clk=True)
    train_set = Dataset(args, args.valid_data_dir, is_clk=True)

    click_model = FCM(mode='dis_sim')

    model_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_vec2'.format(args.timestamp, args.algo, args.batch_size,
        args.lr, args.l2_norm, args.emb_dim, args.keep_prob, args.hidd_size, args.expert_num, args.n_head)
    # if not os.path.exists('{}/log/{}'.format(parse.save_dir, data_set_name)):
    #     os.makedirs('{}/log/{}'.format(parse.save_dir, data_set_name))
    if not os.path.exists('{}/reranker/{}/'.format(args.save_dir, model_name)):
        os.makedirs('{}/reranker/{}/'.format(args.save_dir, model_name))
    save_path = '{}/reranker/{}/ckpt'.format(args.save_dir, model_name)
    metrics_save_path = '{}/reranker/{}.metrics'.format(args.save_dir, model_name)


    training_monitor = {
        'train_loss': [],
        'vali_loss': [],
        'ctr': [],
    }

    # gpu settings
    gpu_options = tf.GPUOptions(allow_growth=True)

    # training process
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        train_losses_step = []

        # before training process
        step = 0
        vali_loss, res = eval(model, test_set, click_model, isrank=False, hist=hist)

        training_monitor['train_loss'].append(None)
        training_monitor['vali_loss'].append(vali_loss)

        # print(vali_loss, res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7][0], res[7][1], res[7][2], res[7][3])
        print("STEP %d  LOSS TRAIN: NULL | LOSS VALI: %.4f  UTILITY: %.4f CTR: %.4f ndcg: %.4f  map: %.4f   "
              "util_row0: %.4f  util_row1: %.4f  util_row2: %.4f  util_row3: %.4f" % (step,
                    vali_loss, res[0], res[1], res[2], res[3], res[4][0], res[4][1], res[4][2], res[4][3]))
        early_stop = False
        batches = train_set.gen_mini_batches(shuffle=True, hist=hist)
        batch_num = len(batches)
        eval_iter_num = batch_num // 4
        best_util = 0
        print('train', batch_num, 'eval iter num', eval_iter_num)

        # begin training process
        for epoch in range(args.epoch_num):
            # if early_stop:
            #     break
            for batch_no in range(batch_num):
                loss = model.train(batches[batch_no])

                step += 1
                train_losses_step.append(loss)

                if step % eval_iter_num == 0:
                    train_loss = sum(train_losses_step) / len(train_losses_step)
                    training_monitor['train_loss'].append(train_loss)
                    train_losses_step = []

                    vali_loss, res = eval(model, test_set, click_model, isrank=True, hist=hist)
                    training_monitor['vali_loss'].append(vali_loss)
                    training_monitor['ctr'].append(res[1])

                    print("EPOCH %d  STEP %d  LOSS TRAIN: %.4f | LOSS VALI: %.4f  UTILITY: %.4f CTR: %.4f ndcg: %.4f  "
                          "map: %.4f  util_col: %.4f  util_row1: %.4f  "
                          "util_row2: %.4f  util_row3: %.4f" % (epoch, step, train_loss, vali_loss, res[0], res[1],
                                res[2], res[3], res[4][0], res[4][1], res[4][2], res[4][3]))
                    if res[1] > best_util:
                        # save model
                        best_util = res[1]
                        model.save(save_path)
                        pkl.dump(res[-1], open(metrics_save_path, 'wb'))
                        print('model saved')
                        # early_stop = False
                        # continue

                    # if len(training_monitor['ctr']) > 2 and epoch > 0:
                    #     if epoch > 40 and best_util - training_monitor['ctr'][-1] > 0.08:
                    #         early_stop = True



def reranker_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_dir', default='../FSR_row/data/CloudTheme/process1/data.train.rank.click')
    parser.add_argument('--test_data_dir', default='../FSR_row/data/CloudTheme/process1/data.test.rank.click')
    parser.add_argument('--valid_data_dir', default='../FSR_row/data/CloudTheme/process1/data.valid.rank.click')
    parser.add_argument('--stat_dir', default='../FSR_row/data/CloudTheme/process1/data.stat')
    parser.add_argument('--save_dir', default='./model/CloudTheme/')
    parser.add_argument('--algo', default='PAR', choices=['PRM', 'DLCM', 'TRNN', 'SetRank', 'GSF', 'DNN', 'miDNN', 'GA', 'DHANR', 'HMoE', 'PAR', ],
                        type=str, help='algorithm name, including PRM, DLCM, SetRank, GSF, miDNN, Seq2Slate, EGR_evaluator, EGR_generator')
    parser.add_argument('--epoch_num', default=500, type=int, help='epochs of each iteration.')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--max_hist_len', default=30, type=int, help='the max length of history')
    parser.add_argument('--lr', default=2e-4, type=float, help='learning rate')
    parser.add_argument('--l2_norm', default=2e-4, type=float, help='l2 loss scale')
    parser.add_argument('--keep_prob', default=0.8, type=float, help='keep probability')
    parser.add_argument('--emb_dim', default=16, type=int, help='size of embedding')
    parser.add_argument('--hidd_size', default=64, type=int, help='hidden size')
    parser.add_argument('--d_model', default=64, type=int, help='input dimension of FFN')
    parser.add_argument('--d_inner_hid', default=128, type=int, help='hidden dimension of FFN')
    parser.add_argument('--n_head', default=4, type=int, help='the number of head in self-attention')
    parser.add_argument('--expert_num', default=12, type=int, help='the number of expert in MMoE')
    parser.add_argument('--hidden_layer', default=[256, 128, 64], type=int, help='size of hidden layer')
    parser.add_argument('--metric_scope', default=[5, 10], type=list, help='the scope of metrics')
    parser.add_argument('--grad_norm', default=0, type=float, help='max norm of gradient')
    parser.add_argument('--decay_steps', default=3000, type=int, help='learning rate decay steps')
    parser.add_argument('--decay_rate', default=1.0, type=float, help='learning rate decay rate')
    parser.add_argument('--timestamp', type=str, default=datetime.datetime.now().strftime("%Y%m%d%H%M"))
    parser.add_argument('--reload_path', type=str, default='', help='model ckpt dir')
    parser.add_argument('--setting_path', type=str, default='', help='setting dir')

    FLAGS, _ = parser.parse_known_args()
    return FLAGS


if __name__ == '__main__':
    # parameters
    random.seed(1234)
    parse = reranker_parse_args()
    print(parse.timestamp)
    if parse.setting_path:
        parse = load_parse_from_json(parse, parse.setting_path)

    train(parse)




