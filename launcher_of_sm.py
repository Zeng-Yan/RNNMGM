import numpy as np
import time
import torch
import torch.utils.data
import torch.nn.utils.rnn as rnn_utils
import torch.optim as optim

import configs as cfg
from utils import tools, smi_tools, file_tools, augmentation
from models import reg_rnn


def is_smi_acceptable(smi, list_of_tokens):
    tokens = smi_tools.split_smi(smi)
    differ = [token for token in tokens if token not in list_of_tokens]
    if differ:
        return False
    else:
        return True


def packer_of_oh_list(data):
    data = sorted(data, key=lambda pairs: len(pairs[0]), reverse=True)
    list_of_oh = [x[0] for x in data]
    y = torch.tensor([x[1] for x in data])
    lengths = [len(oh) for oh in list_of_oh]
    padded = rnn_utils.pad_sequence(list_of_oh, batch_first=True)
    return padded, lengths, y


def predict(m, norm, x):
    y_predict = reg_rnn.predict(m, x)
    y_predict = y_predict.squeeze().detach().cpu().numpy()

    y_predict = norm.recovery(y_predict)

    return y_predict


def train_predictor(path_of_data: str, path_of_pretrained: str, idx_prop=1, epochs=100, k=10, enum_smi=100, patience=5, fq_saving=5):

    records = {'n_train': [], 'n_test': [], 'n_augment': [], 'mae': [], 'rmse': [], 'r2': []}
    time_start = time.time()

    data, head = file_tools.load_data_from_csv(path_of_data, with_head=True)
    lst_smi = [x[0] for x in data]
    lst_tokens = smi_tools.get_list_of_tokens([cfg.BOS + smi + cfg.EOS for smi in lst_smi],
                                              single_split=cfg.SINGLE_TOKENIZE)
    print(f'Tokens: {lst_tokens}')

    name_prop = head[idx_prop]
    print(f'Property name: {name_prop}')
    lst_prop = [float(x[idx_prop]) for x in data]

    fold_idx_sets = tools.stratified_split(lst_prop, k)
    folds = []
    for i in range(k):
        test_idx = fold_idx_sets[i]
        train_idx = [idx for idx in range(len(lst_prop)) if idx not in test_idx]
        train_smi, train_y = [lst_smi[idx] for idx in train_idx], [lst_prop[idx] for idx in train_idx]
        test_smi, test_y = [lst_smi[idx] for idx in test_idx], [lst_prop[idx] for idx in test_idx]

        print(f'Fold {i}, Number of train: {len(train_idx)}, Number of test: {len(test_idx)}')
        lst_train_tokens = smi_tools.get_list_of_tokens([cfg.BOS + smi + cfg.EOS for smi in train_smi],
                                                        single_split=cfg.SINGLE_TOKENIZE)
        print(f'Tokens: {lst_train_tokens}')
        assert lst_tokens == lst_train_tokens, f'[ERROR] Fold-{i} doesnt cover all tokens!'

        folds.append([train_smi, train_y, test_smi, test_y])

    for i in range(k):
        train_smi, train_y, test_smi, test_y = folds[i]
        records['n_train'].append(len(train_smi))
        records['n_test'].append(len(test_smi))

        norm = tools.ZeroSoreNorm(train_y)
        train_y, test_y = norm.norm(train_y), norm.norm(test_y)

        if enum_smi > 0:
            print('Start augmentation ...')
            print(f'Max enumeration times: {enum_smi}.')
            augmented_smi, augmented_y = augmentation.augmentation_by_enum(train_smi, train_y, max_times=enum_smi)
            remain_idx = [idx for idx, smi in enumerate(augmented_smi) if is_smi_acceptable(smi, lst_tokens)]
            train_smi = [augmented_smi[idx] for idx in remain_idx]
            train_y = [augmented_y[idx] for idx in remain_idx]
            print(f'Augmentation finished. {len(augmented_smi)} augmented, {len(train_smi)} accepted.')

        records['n_augment'].append(len(train_smi))
        train_oh = [smi_tools.smi2oh(smi, lst_tokens) for smi in train_smi]
        test_oh = [smi_tools.smi2oh(smi, lst_tokens) for smi in test_smi]

        merge_data = list(zip(train_oh, train_y))
        loader = torch.utils.data.DataLoader(merge_data, batch_size=cfg.REG_BATCH_SIZE, shuffle=True,
                                             collate_fn=packer_of_oh_list)

        m = reg_rnn.RNNPredictor(size_of_oh=len(lst_tokens),
                                 layers_of_rnn=cfg.REG_N_RNN_LAYERS, units_of_rnn=cfg.REG_N_RNN_UNITS,
                                 units_of_nn=cfg.REG_N_NN_UNITS, activation=cfg.REG_NN_ACTIVATION,
                                 dropout_rate=cfg.REG_DROPOUT_RATE)
        m.to(cfg.DEVICE)

        opt = optim.Adam(m.parameters(), lr=cfg.REG_LR_RATE, weight_decay=0.01)

        if path_of_pretrained:
            pre_trained_model = torch.load(path_of_pretrained)['model']
            model_dict = m.state_dict()
            state_dict = {k: v for k, v in pre_trained_model.items() if k in model_dict.keys()}
            print(state_dict.keys())
            model_dict.update(state_dict)
            m.load_state_dict(model_dict)
            print('Pre-trained parameters loaded')

        name_of_m = f'reg_pre_{i}' if path_of_pretrained else f'reg_non_{i}'
        losses = reg_rnn.train(model=m, optimizer=opt, train_loader=loader, valid_x=test_oh, valid_y=test_y,
                               epochs=epochs, fq_of_save=fq_saving, patience=patience, name=name_of_m)

        y_pred_train = predict(m, norm, train_oh)
        y_train = norm.recovery(np.array(train_y))

        y_pred_test = predict(m, norm, test_oh)
        y_test = norm.recovery(np.array(test_y))

        results = list(zip(train_smi, y_train, y_pred_train, [0] * len(y_train)))
        results += list(zip(test_smi, y_test, y_pred_test, [1] * len(y_train)))

        print([tools.mse(y_train, y_pred_train), tools.r_square(y_train, y_pred_train),
               tools.mse(y_test, y_pred_test), tools.r_square(y_test, y_pred_test)])
        print()

        records['mae'].append(np.mean(np.abs(y_pred_test - y_test)))
        records['rmse'].append(np.sqrt(np.mean((y_test - y_pred_test) ** 2)))
        records['r2'].append(tools.r_square(y_test, y_pred_test))

    print('mae', sum(records['mae'])/k)
    print('rmse', sum(records['rmse'])/k)
    print('r2', sum(records['r2'])/k)

    time_end = time.time()
    print(f'Time cost: {time_end-time_start}')

    return 0


def score(p_ori_data, p_data, p_saving, p_m, idx_smi_1, idx_smi_2, idx_prop):
    data, head = file_tools.load_data_from_csv(p_ori_data, with_head=True)
    lst_smi = [x[idx_smi_1] for x in data]
    lst_tokens = smi_tools.get_list_of_tokens([cfg.BOS + smi + cfg.EOS for smi in lst_smi],
                                                  single_split=cfg.SINGLE_TOKENIZE)
    print(lst_tokens, '\n\n')

    name_of_p = head[idx_prop]
    list_of_prop = [float(x[idx_prop]) for x in data]
    norm = tools.ZeroSoreNorm(list_of_prop)
    print(name_of_p, norm.avg, norm.std)

    data, head = file_tools.load_data_from_csv(p_data, with_head=True)
    list_of_smi = [x[idx_smi_2] for x in data]
    list_of_smi = [smi for smi in list_of_smi if is_smi_acceptable(smi, lst_tokens)]

    oh = [smi_tools.smi2oh(smi, lst_tokens) for smi in list_of_smi]

    one_hot_size = len(lst_tokens)
    m = reg_rnn.RNNPredictor(size_of_oh=one_hot_size,
                             layers_of_rnn=cfg.REG_N_RNN_LAYERS, units_of_rnn=cfg.REG_N_RNN_UNITS,
                             units_of_nn=cfg.REG_N_NN_UNITS, activation=cfg.REG_NN_ACTIVATION,
                             dropout_rate=cfg.REG_DROPOUT_RATE)
    m.to(cfg.DEVICE)
    print(m)
    count_of_weights = sum(p.numel() for p in m.parameters())
    print('Number of parameters: {}'.format(count_of_weights))

    trained = torch.load(p_m)['model']
    m.load_state_dict(trained)

    y_predict = reg_rnn.predict(m, oh)
    y_predict = y_predict.squeeze().detach().cpu().numpy()
    y_predict = norm.recovery(y_predict)

    data = [[list_of_smi[idx], y_predict[idx]] for idx in range(len(list_of_smi))]
    file_tools.save_data_to_csv(p_saving, data, head=['smiles', name_of_p])


if __name__ == '__main__':
    train_predictor(path_of_data='data/Dm.csv', path_of_pretrained='record/pt/pt-0005-24.5050.pth',
                    idx_prop=11, epochs=100, k=10, enum_smi=100, patience=3)
