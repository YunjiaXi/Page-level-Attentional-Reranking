import numpy as np
import pickle
import json


class Dataset(object):
    def __init__(self, args, data_path, is_clk=False):
        self.args = args
        # self.gpu_num = args.gpu_num
        self.max_hist_len = args.max_hist_len
        with open(args.stat_dir, 'r') as f:
            stat = json.load(f)

        self.list_num = stat['list_num']
        self.list_len = stat['list_len']
        self.itm_spar_fnum = stat['itm_spar_fnum']
        self.itm_dens_fnum = stat['itm_dens_fnum']
        self.usr_fnum = stat['usr_fnum']
        self.hist_fnum = stat['hist_fnum']
        self.is_clk = is_clk

        self.data_path = data_path
        self.data = self.load_data_from_file(data_path)


    def load_data_from_file(self, file):
        if self.is_clk:
            uid, spar_ft, dens_ft, lbs, clks, hist, hist_len, div_dicts = pickle.load(open(file, 'rb'))
        else:
            uid, spar_ft, dens_ft, lbs, hist, hist_len, div_dicts = pickle.load(open(file, 'rb'))
        usr_ft = np.array(uid).reshape([-1, self.usr_fnum])
        spar_ft = np.array(spar_ft).reshape([-1, self.list_num, self.list_len, self.itm_spar_fnum])
        dens_ft = np.array(dens_ft).reshape([-1, self.list_num, self.list_len, self.itm_dens_fnum])
        lbs = np.array(lbs).reshape([-1, self.list_num, self.list_len])
        print('lbs', lbs.shape, 'spar ft', spar_ft.shape, 'dens ft', dens_ft.shape)

        data_num = lbs.shape[0]

        usr_ft = usr_ft[:data_num]
        spar_ft = spar_ft[:data_num]
        dens_ft = dens_ft[:data_num]
        if self.is_clk:
            clks = np.array(clks).reshape([-1, self.list_num, self.list_len])[: data_num]
            print('clk', clks.shape)
            print('average click per page', np.sum(clks)/data_num)
            print('average lb per page', np.sum(lbs)/data_num)


        length = np.ones([data_num, self.list_num], dtype=np.int32) * self.list_len
        mask = np.ones([data_num, self.list_num, self.list_len])

        hist_list = []
        print('hist num', len(hist))
        hist_len = list(hist_len)
        if self.max_hist_len:
            for i, lst in enumerate(hist[:data_num]):
                if len(lst) >= self.max_hist_len:
                    new_hist = lst[-self.max_hist_len:]
                    hist_len[i] = self.max_hist_len
                    new_hist = np.array(new_hist)
                else:
                    lst = np.array(lst)
                    new_hist = np.pad(lst, (0, self.max_hist_len - len(lst)), 'constant')
                hist_list.append(new_hist)
            hist_list = np.reshape(np.array(hist_list), [-1, self.max_hist_len, self.hist_fnum])
            hist_len = np.array(hist_len)

        data = {'usr_ft': usr_ft,
                'mask': mask,
                'lb': lbs,
                'spar_ft': spar_ft,
                'dens_ft': dens_ft,
                'hist_ft': hist_list, 'hist_len': hist_len,
                'len': length,
                'div_dict': div_dicts,
                }
        if self.is_clk:
            data['clk'] = clks
        # print('click num of is {}'.format(click_num))
        return data

    def _one_mini_batch(self, data, indices, hist=False):
        """
        Get one mini batch ranking 0
        """
        # list_rand = np.random.randint(self.list_len, size=self.list_len)

        batch_data = {
            'usr_ft': data['usr_ft'][indices],
            'spar_ft': data['spar_ft'][indices],
            'dens_ft': data['dens_ft'][indices],
            'len': data['len'][indices],
            'mask': data['mask'][indices],
            'div_dict': data['div_dict']
            }
        if hist:
            batch_data['hist_ft'] = data['hist_ft'][indices]
            batch_data['hist_len'] = data['hist_len'][indices]
        if self.is_clk:
            batch_data['clk'] = data['clk'][indices]
            batch_data['lb'] = data['lb'][indices]
        else:
            batch_data['clk'] = data['lb'][indices]

        return batch_data

    def gen_mini_batches(self, shuffle=True, hist=False):

        data_size = self.data['usr_ft'].shape[0]

        indices = np.arange(data_size // self.args.batch_size * self.args.batch_size)
        if shuffle:
            np.random.shuffle(indices)
        indices = indices.tolist()
        batches = []

        # for 0 parallel in multi-gpu cases
        # indices += indices[:(self.gpu_num - data_size % self.gpu_num) % self.gpu_num]
        for batch_start in np.arange(0, len(list(indices)), self.args.batch_size):
            batch_indices = indices[batch_start: batch_start + self.args.batch_size]
            batches.append(self._one_mini_batch(self.data, batch_indices, hist=hist))
        return batches
