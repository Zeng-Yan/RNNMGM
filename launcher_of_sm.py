import numpy as np
import time
import torch
import torch.optim as optim

import configs as cfg
from utils import builder, tools, smi_tools, augmentation
from models import reg_rnn


def train_predictor(data_path: str, pretrained_path: str, prop_idx=1, epochs=100, k=10, enum_smi=100, patience=5, fq_saving=5):
    """

    :param data_path:
    :param pretrained_path:
    :param prop_idx:
    :param epochs:
    :param k:
    :param enum_smi:
    :param patience:
    :param fq_saving:
    :return:
    """

    records = {'n_train': [], 'n_test': [], 'n_augment': [], 'mae': [], 'rmse': [], 'r2': []}
    time_start = time.time()

    # 加载数据
    data, head = tools.load_data_from_csv(data_path, with_head=True)  # 从文件加载数据
    smiles = [x[0] for x in data]  # 获取smiles
    tokens = smi_tools.gather_tokens([cfg.BOS + smi + cfg.EOS for smi in smiles], single_split=cfg.SINGLE_TOKENIZE)  # 获得token列表
    print(f'Tokens: {tokens}')
    print(f'Property name: {head[prop_idx]}')
    lst_prop = [float(x[prop_idx]) for x in data]  # 获取标签数据

    # 划分数据集
    folds = tools.train_test_split(smiles, lst_prop, k)
    assert smi_tools.if_each_fold_cover_all_tokens(folds, tokens), f'[ERROR] Not every fold cover all tokens!'

    for i in range(k):
        train_smi, train_y, test_smi, test_y = folds[i]
        norm = tools.ZeroSoreNorm(train_y)
        train_y, test_y = norm.norm(train_y), norm.norm(test_y)  # 数据标准化

        records['n_train'].append(len(train_smi))
        records['n_test'].append(len(test_smi))

        if enum_smi > 0:
            print('Start augmentation ...')
            print(f'Max enumeration times: {enum_smi}.')
            augmented_smi, augmented_y = augmentation.augmentation_by_enum(train_smi, train_y, max_times=enum_smi)
            remain_idx = [idx for idx, smi in enumerate(augmented_smi) if smi_tools.if_oov_exclude(smi, tokens)]
            train_smi = [augmented_smi[idx] for idx in remain_idx]  # 去掉增强后smiles可能包含多余token的数据
            train_y = [augmented_y[idx] for idx in remain_idx]
            print(f'Augmentation finished. {len(augmented_smi)} augmented, {len(train_smi)} accepted.')

        records['n_augment'].append(len(train_smi))
        train_oh = [smi_tools.smiles2tensor(smi, tokens) for smi in train_smi]  # 独热编码
        test_oh = [smi_tools.smiles2tensor(smi, tokens) for smi in test_smi]

        loader = builder.reg_packer(train_oh, train_y)  # 加载训练数据

        # 初始化
        m = builder.build_reg(len(tokens), pretrained_path, True if pretrained_path else False)
        opt = optim.Adam(m.parameters(), lr=cfg.REG_LR_RATE, weight_decay=0.01)  # 初始化优化器

        # 训练模型
        name_of_m = f'reg_pre_{i}' if pretrained_path else f'reg_non_{i}'
        losses = reg_rnn.train(model=m, optimizer=opt, train_loader=loader, valid_x=test_oh, valid_y=test_y,
                               epochs=epochs, fq_of_save=fq_saving, patience=patience, name=name_of_m)

        tools.save_data_to_csv(f'record/REG_{enum_smi}_FOLD-{i}_LOSS.csv', losses, ['epoch', 'train_loss', 'valid_loss'])

        y_pred_train = builder.normed_predict(train_oh, m, norm)  # 训练数据表现
        y_train = norm.recovery(np.array(train_y))

        y_pred_test = builder.normed_predict(test_oh, m, norm)  # 测试数据表现
        y_test = norm.recovery(np.array(test_y))

        results = list(zip(train_smi, y_train, y_pred_train, [0] * len(y_train)))
        results += list(zip(test_smi, y_test, y_pred_test, [1] * len(y_test)))
        tools.save_data_to_csv(f'record/REG_{enum_smi}_FOLD-{i}_RESULTS.csv', results, ['smi', 'y', 'y_pred', 'test'])

        # 打印
        print([tools.mse(y_train, y_pred_train), tools.r_square(y_train, y_pred_train),
               tools.mse(y_test, y_pred_test), tools.r_square(y_test, y_pred_test)])
        print()

        records['mae'].append(np.mean(np.abs(y_pred_test - y_test)))
        records['rmse'].append(np.sqrt(np.mean((y_test - y_pred_test) ** 2)))
        records['r2'].append(tools.r_square(y_test, y_pred_test))

    rcd = list(np.array(list(records.values())).T)
    tools.save_data_to_csv(f'record/REG_{enum_smi}_Performance.csv', rcd,
                                ['train', 'test', 'augment', 'mae', 'rmse', 'r2'])
    print('mae', sum(records['mae'])/k)
    print('rmse', sum(records['rmse'])/k)
    print('r2', sum(records['r2'])/k)

    time_end = time.time()
    print(f'Time cost: {time_end-time_start}')

    return 0


def score(train_data_path: str, data_path: str, model_path: str, saving_path: str,
          smi_idx_1: int, smi_idx_2: int, prop_idx: int):

    # 获取token
    data, head = tools.load_data_from_csv(train_data_path, with_head=True)
    smiles = [x[smi_idx_1] for x in data]
    tokens = smi_tools.gather_tokens([cfg.BOS + smi + cfg.EOS for smi in smiles], single_split=cfg.SINGLE_TOKENIZE)  # 获得token列表
    print(tokens, '\n\n')

    list_of_prop = [float(x[prop_idx]) for x in data]
    norm = tools.ZeroSoreNorm(list_of_prop)
    print(head[prop_idx], norm.avg, norm.std)

    # 获取数据
    data, _ = tools.load_data_from_csv(data_path, with_head=True)
    smiles = [x[smi_idx_2] for x in data]
    raw_smiles = [smi for smi in smiles if smi_tools.if_oov_exclude(smi, tokens)]  # 过滤
    smiles = [smi for smi in raw_smiles]

    # 初始化模型
    m = builder.build_reg(len(tokens), model_path, False)
    y_predict = builder.smiles_predict(smiles, tokens, m, norm)

    # 保存
    data = [[raw_smiles[idx], y_predict[idx]] for idx in range(len(raw_smiles))]
    tools.save_data_to_csv(saving_path, data, head=['smiles', head[prop_idx]])


if __name__ == '__main__':
    train_predictor('data/Dm.csv', '',
                    prop_idx=11, epochs=100, k=10, enum_smi=100, patience=3, fq_saving=5)
