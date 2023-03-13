import numpy as np
import pickle


class FCM(object):
    def __init__(self, mode='dis_sim', dist=1):
        self.mode = mode
        self.dist = dist
        self.list_num = 4
        self.list_len = 10
        self.get_row_exam()

    def get_row_exam(self):
        row_exam = 1 / np.arange(1, self.list_len + 1) ** 0.5
        col_exam = 1 / np.arange(1, self.list_num + 1) ** 0.4
        row_exam = np.tile(np.reshape(row_exam, [1, -1]), [self.list_num, 1])
        col_exam = np.tile(np.reshape(col_exam, [-1, 1]), [1, self.list_len])
        self.exam = row_exam * col_exam

    def get_distance(self, item1, item2):
        sim = np.where(item1 == item2, 1, 0)
        return np.mean(sim)

    def get_sim_prob(self, itm1, itm2):

        sim = np.dot(itm1, itm2) / (np.linalg.norm(itm1, axis=-1) * np.linalg.norm(itm2, axis=-1))

        if self.mode == 'sim':
            prob = sim
        else:
            prob = (1 - sim)/2 + 0.5
        return prob

    def get_neighbor(self, i, j, itms):
        candidate_pos = [(i-1, j), (i, j-1), (i+1, j), (i, j+1)]
        neighbors = []
        for ii, jj in candidate_pos:
            if ii >= 0 and ii < self.list_num and jj >= 0 and jj < self.list_len:
                neighbors.append(itms[ii][jj])
        return neighbors

    def get_cmp_prob(self, cur_itm, neighbors, prev_clk_itm):
        mean_neig = np.mean(np.array(neighbors), axis=0)
        return self.get_sim_prob(mean_neig, cur_itm)

    def generate_page_click(self, itms, lbs, sim_dict):
        orig_shape = lbs.shape
        itms = np.reshape(itms, [self.list_num, self.list_len, -1])
        lbs = np.reshape(lbs, [self.list_num, self.list_len])

        prev_clk_itm = np.zeros([128])
        clk_prob, clk = [], []
        for i in range(self.list_num):
            list_click, list_click_prob = [], []
            for j in range(self.list_len):
                rel = lbs[i][j]
                if rel:
                    if sum(itms[i][j]) == 0:
                        print(i, j, lbs[i][j], sum(itms[i][j]))
                    cur_itm = itms[i][j]
                    neighbor = self.get_neighbor(i, j, itms)
                    cmp = self.get_cmp_prob(cur_itm, neighbor, prev_clk_itm)
                    click_prob = self.exam[i][j] * cmp
                    click = 1 if np.random.rand() < click_prob else 0
                    if click:
                        prev_clk_itm = cur_itm
                    list_click.append(click)
                    list_click_prob.append(click_prob)
                else:
                    list_click.append(0)
                    list_click_prob.append(0)
            clk.append(list_click)
            clk_prob.append(list_click_prob)

        clk = np.array(clk).reshape(orig_shape)
        clk_prob = np.array(clk_prob).reshape(orig_shape)

        return clk.tolist(), clk_prob.tolist()


