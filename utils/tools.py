import numpy as np


def stratified_split(y, k: int) -> tuple:
    from sklearn.model_selection import StratifiedKFold
    y_max, y_min = max(y), min(y)
    y_gap = (y_max - y_min) / k
    y_hash = [int((i - y_min) // y_gap) for i in y]  # hash y 作为样本的类别标签

    # StratifiedKFold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=2)
    idx_sets = []
    for idx_train, idx_test in skf.split(y, y_hash):  # skf.split划分样本为k个fold，返回每个fold的train和test的索引。
        idx_sets.append(idx_test)
        # idx_sets.append(list(np.concatenate((idx_train, idx_test), axis=0)))
    return idx_sets


def mse(y_real, y_predict):
    return np.mean((y_real - y_predict) ** 2)


def r_square(y_real, y_predict):
    """
    计算真实值与预测值之间的r2
    :param y_real: 真实值；
    :param y_predict: 预测值；
    :return:
    """

    y_real = np.array(y_real)
    y_predict = np.array(y_predict)

    a = np.sum(np.square(y_predict - y_real))
    b = np.sum(np.square(y_real - np.mean(y_real)))
    return 1 - a / b


class ZeroSoreNorm:

    def __init__(self, values: np.array):
        self.avg = np.average(values)
        self.std = np.std(values)

    def norm(self, values: np.array):
        values = np.array(values)
        return (values - self.avg) / self.std

    def recovery(self, values: np.array):
        values = np.array(values)
        return values * self.std + self.avg


