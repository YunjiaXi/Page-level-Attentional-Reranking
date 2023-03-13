import pickle

import numpy as np
import json
import time
from click_model import FCM


def normalize(v):
    v = np.array(v)
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


def load_parse_from_json(parse, setting_path):
    with open(setting_path, 'r') as f:
        setting = json.load(f)
    parse_dict = vars(parse)
    for k, v in setting.items():
        parse_dict[k] = v
    return parse


def save_file(data, save_file):
    with open(save_file, 'w') as f:
        for v in data:
            line = '\t'.join([str(i) for i in v]) + '\n'
            f.write(line)


def eval(model, dataset, click_model, isrank, hist=False, _print=False):
    losses = []

    batches = dataset.gen_mini_batches(shuffle=True, hist=hist)
    batch_num = len(batches)
    print('eval', dataset.args.batch_size, batch_num)

    t = time.time()
    page_preds = [[] for _ in range(dataset.list_num)]

    for batch_no in range(batch_num):
        preds, labels, loss = model.eval(batches[batch_no])
        for i, pred in enumerate(preds):
            page_preds[i].extend(pred)
        losses.append(loss)
    page_preds = np.array(page_preds).transpose(1, 0, 2)

    loss = sum(losses) / len(losses)
    # print('finish eval')

    res = evaluate_fcm(batches, click_model, page_preds, isrank, _print)

    print("EVAL TIME: %.4fs" % (time.time() - t))
    # return loss, res_low, res_high
    return loss, res

#
# def save_rank(model, dataset):
#     batches = dataset.gen_mini_batches(shuffle=False, hist=False)
#     batch_num = len(batches)
#     # print('eval', batch_num)
#     page_preds = [[] for _ in range(dataset.list_num)]
#
#     for batch_no in range(batch_num):
#         preds, labels, loss = model.eval(batches[batch_no])
#         for i, pred in enumerate(preds):
#             page_preds[i].extend(pred)
#
#     preds = np.array(page_preds).transpose(1, 0, 2).tolist()
#     rank_click(dataset.data_path, preds, dataset.data_path+'.rank.click')
#
#
# def rank_click(data_file, preds, out_file):
#     data = pickle.load(open(data_file, 'rb'))
#     # sim_dicts = pickle.load(open(data_file, 'rb'))[-1]
#     click_model = FCM(mode='dis_sim')
#
#     uid, spars, denss, lbs, hist, hist_len, div_dicts = data
#     out_spars, out_denss, out_lbs, out_clks = [], [], [], []
#     for per_spar, per_dens, per_lbs, per_preds, div_dict in zip(spars, denss, lbs, preds, div_dicts):
#         out_spar, out_dens, out_lb = [], [], []
#         for spar, dens, lb, pred in zip(per_spar, per_dens, per_lbs, per_preds):
#             rerank_idx = sorted(list(range(len(pred))), key=lambda k: pred[k], reverse=True)
#             out_spar.append(np.array(spar)[rerank_idx].tolist())
#             out_dens.append(np.array(dens)[rerank_idx].tolist())
#             out_lb.append(np.array(lb)[rerank_idx].tolist())
#         out_spars.append(out_spar)
#         out_denss.append(out_dens)
#         out_lbs.append(out_lb)
#
#         clk, clk_prob = click_model.generate_page_click(np.array(out_dens),
#                                          np.array(out_lb), div_dict)
#         out_clks.append(clk)
#
#     with open(out_file, 'wb') as f:
#         pickle.dump([uid, out_spars, out_denss, out_lbs, out_clks, hist, hist_len, div_dicts], f)
#
#
# def rank(data_file, col_preds, row_preds, out_file):
#     data = pickle.load(open(data_file, 'rb'))
#
#     uid, cols, rows, col_lbs, row_lbs, hist, hist_len, div_dict = data
#     out_cols, out_rows, out_col_lbs, out_row_lbs = [], [], [], []
#     for per_cols, per_rows, per_col_lbs, per_row_lbs, per_col_preds, per_row_preds in zip(cols, rows, col_lbs, row_lbs, col_preds, row_preds):
#         out_col, out_col_lb = [], []
#         for col, col_lb, col_pred in zip(per_cols, per_col_lbs, per_col_preds):
#             rerank_idx = sorted(list(range(len(col_pred))), key=lambda k: col_pred[k], reverse=True)
#             out_col.append(np.array(col)[rerank_idx].tolist())
#             out_col_lb.append(np.array(col_lb)[rerank_idx].tolist())
#         out_cols.append(out_col)
#         out_col_lbs.append(out_col_lb)
#
#         out_row, out_row_lb = [], []
#         for row, row_lb, row_pred in zip(per_rows, per_row_lbs, per_row_preds):
#             rerank_idx = sorted(list(range(len(row_pred))), key=lambda k: row_pred[k], reverse=True)
#             out_row.append(np.array(row)[rerank_idx].tolist())
#             out_row_lb.append(np.array(row_lb)[rerank_idx].tolist())
#         out_rows.append(out_row)
#         out_row_lbs.append(out_row_lb)
#
#     # print('col', np.array(out_cols).shape)
#     # print('row', np.array(out_rows).shape)
#
#     with open(out_file, 'wb') as f:
#         pickle.dump([uid, out_cols, out_rows, out_col_lbs, out_row_lbs, hist, hist_len, div_dict], f)
#


def get_reranked_batch_ft(fts, lbs, preds):
    # rerank_cols, rerank_rows = [], []
    rerank_fts, rerank_lbs = [], []
    for per_fts, per_lbs, per_preds in zip(fts, lbs, preds):
        rerank_per_ft, rerank_per_lb = [], []
        for ft, lb, pred in zip(per_fts, per_lbs, per_preds):
            rerank_idx = sorted(list(range(len(pred))), key=lambda k: pred[k], reverse=True)
            rerank_per_ft.append(np.array(ft)[rerank_idx])
            rerank_per_lb.append(np.array(lb)[rerank_idx])
        rerank_fts.append(rerank_per_ft)
        rerank_lbs.append(rerank_per_lb)

    return np.array(rerank_fts), np.array(rerank_lbs)


def batch_rel_metrics(batch_lbs, batch_preds, is_rank, scopes):
    ndcg, map = [[[] for j in range(len(scopes[i]))] for i in range(len(scopes))], [[[] for j in range(len(scopes[i]))] for i in range(len(scopes))]
    for preds, lbs in zip(batch_preds, batch_lbs):
        preds, lbs = preds.tolist(), lbs.tolist()
        for idx, pred, label, scope in zip(range(len(preds)), preds, lbs, scopes):
            if is_rank:
                final = sorted(range(len(pred)), key=lambda k: pred[k], reverse=True)
            else:
                final = list(range(len(pred)))
            click = np.array(label)[final].tolist()  # reranked labels
            gold = sorted(range(len(label)), key=lambda k: label[k], reverse=True)  # optimal list for ndcg

            for i, scop in enumerate(scope):
                ideal_dcg, dcg, de_dcg, de_idcg, AP_value, AP_count, util = 0, 0, 0, 0, 0, 0, 0
                cur_scope = min(scop, len(label))
                for _i, _g, _f in zip(range(1, cur_scope + 1), gold[:cur_scope], final[:cur_scope]):
                    dcg += (pow(2, click[_i - 1]) - 1) / (np.log2(_i + 1))
                    ideal_dcg += (pow(2, label[_g]) - 1) / (np.log2(_i + 1))

                    if click[_i - 1] >= 1:
                        AP_count += 1
                        AP_value += AP_count / _i

                _ndcg = float(dcg) / ideal_dcg if ideal_dcg != 0 else 0.
                _map = float(AP_value) / AP_count if AP_count != 0 else 0.
                ndcg[idx][i].append(_ndcg)
                map[idx][i].append(_map)
    return np.array(ndcg), np.array(map)


def evaluate_fcm(batches, click_model, page_preds, is_rank, _print=False):
    print('eval fcm')
    t = time.time()
    list_num = page_preds.shape[1]
    scopes = [[10], [10], [10], [10]]
    batch_num = len(batches)
    batch_page_preds = np.split(page_preds, batch_num, axis=0)

    clks, clk_probs = [], []
    for i in range(batch_num):
        itms = batches[i]['dens_ft']
        lbs = batches[i]['lb']

        if is_rank:
            itms, lbs = get_reranked_batch_ft(itms, lbs, batch_page_preds[i])
        for itm, lb, div_dict in zip(itms, lbs, batches[i]['div_dict']):
            clk, clk_prob = click_model.generate_page_click(itm, lb, div_dict)
            clks.append(clk)
            clk_probs.append(clk_prob)

    util = np.sum(np.array(clks), axis=-1)

    clk_probs = np.sum(np.array(clk_probs), axis=-1)
    print("generate click TIME: %.4fs" % (time.time() - t))
    t = time.time()

    ndcg, map, div = [], [], []

    for i in range(batch_num):
        batch_ndcg, batch_map = batch_rel_metrics(batches[i]['lb'], batch_page_preds[i], is_rank, scopes)
        ndcg.append(batch_ndcg)
        map.append(batch_map)

    ndcg = np.concatenate(ndcg, axis=-1)
    map = np.concatenate(map, axis=-1)
    print("rel TIME: %.4fs" % (time.time() - t))

    mean_util_per_list = np.mean(clk_probs, axis=0)
    mean_util = np.mean(np.sum(util, axis=-1))
    mean_clk_prob = np.mean(np.sum(clk_probs, axis=-1))
    mean_ndcg = non_zero_mean(ndcg)
    mean_map = non_zero_mean(map)

    return mean_util, mean_clk_prob, mean_ndcg, mean_map, mean_util_per_list.tolist(), [util, clk_probs, ndcg, map]


def non_zero_mean(arr):
    exist = (arr != 0)
    num = np.sum(arr)
    # if num == 0:
    #     return 0
    den = np.sum(exist)
    return num/den

