
from collections import defaultdict
import numpy as np
from scipy import optimize


class TaylorLowRegressor():
    def __init__(self):
        pass

    def fit(self, word_seq, width):
        word2posit = defaultdict(lambda x: [])
        for i, word in enumerate(word_seq):
            word2posit[word].append(i)

        word_list = []
        mu_list = []
        sigma_list = []
        freq_list = []
        posit_list = []
        # FIXME: slow
        for i, w, posits in enumerate(word2posit.items()):
            freqs = []
            for j in range(len(word_seq) // width + 1):
                start_idx = j * width
                end_idx = (j+1) * width
                freqs.append(len(np.where((posits >= start_idx) & (posits < end_idx))[0]))
            word_list.append(w)
            mu_list.append(np.mean(freqs))
            sigma_list.append(np.std(freqs))
            freq_list.append(np.sum(freqs))
            posit_list.append(posits)


        res = self._run_optim(mu_list, sigma_list)
        error = np.sqrt(np.mean(np.power(taylor_objective(res[0], mu_list, sigma_list), 2)))

        self.res = res
        self.error = error
        self.stats = {"word": word_list,
                      "mu": mu_list,
                      "sigma": sigma_list,
                      "freq": freq_list,
                      "position": posit_list}


    def _run_optim(self, xs, ys):
        assert len(xs) == len(ys), "Inconsistent length"
        init_param = np.array([1.0, 0.0])
        return optimize.leastsq(taylor_objective, x0=init_param,
                                args=(list(xs), list(ys)))


def taylor_objective(param, x, y):
    return np.log10(y) - np.log10(param[0] * x ** param[1])
