import torch
import torch.utils.data
import torch.optim as optim

import configs as cfg
from utils import file_tools, smi_tools, tools
from models import clm_rnn


def smi_packer(list_of_smi, list_of_tokens):
    list_of_tokenized_smi = [smi_tools.smi2tokens(smi, list_of_tokens) for smi in list_of_smi]
    list_of_length = [len(smi) for smi in list_of_tokenized_smi]
    max_length = max(list_of_length)

    mat_of_ohe, mat_of_lb = smi_tools.smi_list2oh_mat(list_of_smi, list_of_tokens, max_length)
    mat_of_ohe_pre = mat_of_ohe[:, 0:max_length - 1, :]
    mat_of_tar_lb = mat_of_lb[:, 1::]

    mat_of_len_mask = [[1 if i < (x - 1) else 0 for i in range(max_length - 1)] for x in
                       list_of_length]
    mat_of_len_mask = torch.tensor(mat_of_len_mask)

    print('Shape of input(one-hot): ', mat_of_ohe_pre.size())
    print('Shape of target(label): ', mat_of_tar_lb.size())

    merge_data = torch.utils.data.TensorDataset(mat_of_ohe_pre, mat_of_tar_lb, mat_of_len_mask)
    loader = torch.utils.data.DataLoader(merge_data, batch_size=cfg.CLM_BATCH_SIZE, shuffle=True)

    return loader


def train_clm(path_of_data: str, smi_idx: int, path_of_m: str, name_of_m: str, epochs=100, lr=cfg.CLM_LR_RATE, fq_saving=5):

    data, head = file_tools.load_data_from_csv(path_of_data, with_head=True)
    list_of_smi = [x[smi_idx] for x in data]
    list_of_smi = [cfg.BOS + smi + cfg.EOS for smi in list_of_smi]
    list_of_tokens = smi_tools.get_list_of_tokens(list_of_smi, single_split=cfg.SINGLE_TOKENIZE)
    print(f'Tokens: {list_of_tokens}')
    loader = smi_packer(list_of_smi, list_of_tokens)

    m = clm_rnn.NNLM(size_of_oh=len(list_of_tokens),
                     layers_of_rnn=cfg.CLM_N_RNN_LAYERS, units_of_rnn=cfg.CLM_N_RNN_UNITS,
                     dropout_rate=cfg.CLM_DROPOUT_RATE)
    m.to(cfg.DEVICE)
    print(f'Structure of model:\n{m}')
    count_of_weights = sum(p.numel() for p in m.parameters())
    print(f'Number of parameters: {count_of_weights}')

    params_of_m = m.parameters()
    if path_of_m:
        print(f'Load pre-trained model: {path_of_m}')
        checkpoint = torch.load(path_of_m)
        m.load_state_dict(checkpoint['model'])
        for name, value in m.lstm.named_parameters():
            # name_of_p = ['weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0']
            name_of_p = []
            if name in name_of_p:
                print(f'[WARNING] Weights fixed: {name}')
                value.requires_grad = False

        params_of_m = filter(lambda p: p.requires_grad, m.parameters())
    else:
        print('No pre-trained model loaded')

    opt = optim.Adam(params_of_m, lr=lr)
    records = clm_rnn.train(model=m, optimizer=opt, data_loader=loader,
                            epochs=epochs, fq_of_save=fq_saving, name=name_of_m)

    return 0


def generate(n: int, idx: int, p_data: str, p_model: str, p_saving: str, limit: int) -> list:
    data, head = file_tools.load_data_from_csv(p_data, with_head=True)
    lst_smi = [x[idx] for x in data]
    lst_smi_ori = [smi_tools.to_canonical_smi(smi) for smi in lst_smi]
    lst_smi_ori = [smi for smi in lst_smi_ori if smi is not None]

    lst_tokens = smi_tools.get_list_of_tokens([cfg.BOS + smi + cfg.EOS for smi in lst_smi],
                                              single_split=cfg.SINGLE_TOKENIZE)
    m = clm_rnn.NNLM(size_of_oh=len(lst_tokens),
                     layers_of_rnn=cfg.CLM_N_RNN_LAYERS, units_of_rnn=cfg.CLM_N_RNN_UNITS,
                     dropout_rate=cfg.CLM_DROPOUT_RATE)
    m.to(cfg.DEVICE)

    checkpoint = torch.load(p_model)
    m.load_state_dict(checkpoint['model'])
    print('Model loaded !')

    print('Sampling ...')
    lst_smi_gen = clm_rnn.sample(model=m, token_list=lst_tokens, number_of_sample=n)
    lst_smi_valid = [smi_tools.to_canonical_smi(smi) for smi in lst_smi_gen]
    lst_smi_valid = [smi for smi in lst_smi_valid if smi is not None]
    lst_smi_unique = list(set(lst_smi_valid))
    lst_smi_novel = [smi for smi in lst_smi_unique if smi not in lst_smi_ori]

    print('Sampling Finished !')
    print(f'Sample:{n}, Valid:{len(lst_smi_valid)}, Unique:{len(lst_smi_unique)}, Novel:{len(lst_smi_novel)}, Limit: {limit}')
    lst_smi_novel = lst_smi_novel[0:limit]
    file_tools.save_data_to_csv(p_saving, [[smi] for smi in lst_smi_novel], ['smiles'])

    return lst_smi_novel


if __name__ == '__main__':
    train_clm(path_of_data='data/Df.csv', smi_idx=0, path_of_m='', name_of_m='pt', epochs=20, fq_saving=1)
    # train_clm(path_of_data='data/Dm.csv', smi_idx=0, path_of_m='record/pt-0005-24.5050.pth',
    #           name_of_m='tl', epochs=40, fq_saving=5)
