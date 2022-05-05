import numpy as np
import csv


def load_data_from_csv(csv_path: str, with_head=True) -> (list, list):

    with open(csv_path, 'r') as f:
        data = csv.reader(f)
        data = list(data)

    if with_head:
        return data[1:], data[0]
    else:
        return data, None


def save_data_to_csv(csv_path: str, data: list, head: list):

    with open(csv_path, 'w+', newline='') as f:
        f_csv = csv.writer(f)
        if head:
            f_csv.writerow(head)
        f_csv.writerows(data)

    return 0


def stratified_split(y, k: int, seed=2) -> list:
    from sklearn.model_selection import StratifiedKFold
    y_max, y_min = max(y), min(y)
    y_gap = (y_max - y_min) / k
    y_hash = [int((i - y_min) // y_gap) for i in y]  # hash y 作为样本的类别标签

    # StratifiedKFold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    idx_sets = []
    for idx_train, idx_test in skf.split(y, y_hash):  # skf.split划分样本为k个fold，返回每个fold的train和test的索引。
        idx_sets.append(idx_test)
        # idx_sets.append(list(np.concatenate((idx_train, idx_test), axis=0)))
    return idx_sets


def train_test_split(x, y, k: int) -> list:
    # 划分数据集
    fold_idx_sets = stratified_split(y, k)
    # 训练集和测试集划分
    folds = []
    for i in range(k):
        test_idx = fold_idx_sets[i]
        train_idx = [idx for idx in range(len(y)) if idx not in test_idx]
        train_x, train_y = [x[idx] for idx in train_idx], [y[idx] for idx in train_idx]
        test_x, test_y = [x[idx] for idx in test_idx], [y[idx] for idx in test_idx]
        folds.append([train_x, train_y, test_x, test_y])
    return folds


def mse(y_real, y_predict) -> np.array:
    return np.mean((y_real - y_predict) ** 2)


def r_square(y_real, y_predict) -> np.array:

    y_real = np.array(y_real)
    y_predict = np.array(y_predict)

    a = np.sum(np.square(y_predict - y_real))
    b = np.sum(np.square(y_real - np.mean(y_real)))
    return 1 - a / b


class ZeroSoreNorm:

    def __init__(self, values: np.array):
        self.avg = np.average(values)
        self.std = np.std(values)

    def norm(self, values: np.array) -> np.array:
        values = np.array(values)
        return (values - self.avg) / self.std

    def recovery(self, values: np.array) -> np.array:
        values = np.array(values)
        return values * self.std + self.avg


