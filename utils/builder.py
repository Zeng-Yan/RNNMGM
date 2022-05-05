import torch
import torch.utils.data
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from models import clm_rnn, reg_rnn
from utils import smi_tools, tools
import configs as cfg


def clm_packer(smiles: list, tokens: list) -> torch.utils.data.DataLoader:

    encoded_smiles = [smi_tools.smiles2tensor(smi, tokens, True) for smi in smiles]
    mat = pad_sequence(encoded_smiles, True, 0)
    max_length = mat.size(1)
    ipt = mat[:, 0:max_length - 1, :]

    labeled_smiles = [smi_tools.smiles2tensor(smi, tokens, False) for smi in smiles]
    mat = pad_sequence(labeled_smiles, True, 0)
    target = mat[:, 1::]

    lengths = [len(smi) for smi in encoded_smiles]
    mask = [[1 if i < (x - 1) else 0 for i in range(max_length - 1)] for x in lengths]  # 长度掩码，seq + EOS，不含BOS
    mask = torch.tensor(mask)

    print('Shape of input(one-hot): ', ipt.size())
    print('Shape of target(label): ', target.size())
    print('Shape of mask: ', mask.size())

    merge_data = torch.utils.data.TensorDataset(ipt, target, mask)
    loader = torch.utils.data.DataLoader(merge_data, batch_size=cfg.CLM_BATCH_SIZE, shuffle=True)
    return loader


def reg_packer(x, y) -> torch.utils.data.DataLoader:

    def collate(data):
        data = sorted(data, key=lambda pairs: len(pairs[0]), reverse=True)

        x = [record[0] for record in data]
        y = torch.tensor([record[1] for record in data])

        lengths = [len(oh) for oh in x]
        padded = pad_sequence(x, batch_first=True)

        return padded, lengths, y

    merge_data = list(zip(x, y))
    loader = torch.utils.data.DataLoader(merge_data, batch_size=cfg.REG_BATCH_SIZE, shuffle=True,
                                         collate_fn=collate)

    return loader


def build_reg(size: int, path: str, pretrained=False):
    # initialize model
    m = reg_rnn.RNNPredictor(size_of_oh=size,
                             layers_of_rnn=cfg.REG_N_RNN_LAYERS, units_of_rnn=cfg.REG_N_RNN_UNITS,
                             units_of_nn=cfg.REG_N_NN_UNITS, activation=cfg.REG_NN_ACTIVATION,
                             dropout_rate=cfg.REG_DROPOUT_RATE)
    m.to(cfg.DEVICE)
    # print model info
    print(f'Structure of model:\n{m}')
    n_weights = sum(p.numel() for p in m.parameters())
    print(f'Number of parameters: {n_weights}')
    # load parameters
    if pretrained:
        print(f'Load pre-trained model: {path}')
        checkpoint = torch.load(path)
        pre_trained_model = checkpoint['model']
        model_dict = m.state_dict()
        state_dict = {k: v for k, v in pre_trained_model.items() if k in model_dict.keys()}
        print(state_dict.keys())  # dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
        model_dict.update(state_dict)
        m.load_state_dict(model_dict)
        print('Pre-trained parameters loaded')
    elif path:
        print(f'Load model: {path}')
        checkpoint = torch.load(path)
        m.load_state_dict(checkpoint['model'])

    return m


def build_clm(size: int, path: str):
    # initialize model
    m = clm_rnn.NNLM(size_of_oh=size,
                     layers_of_rnn=cfg.CLM_N_RNN_LAYERS, units_of_rnn=cfg.CLM_N_RNN_UNITS,
                     dropout_rate=cfg.CLM_DROPOUT_RATE)
    m.to(cfg.DEVICE)
    # print model info
    print(f'Structure of model:\n{m}')
    n_weights = sum(p.numel() for p in m.parameters())
    print(f'Number of parameters: {n_weights}')
    # load parameters
    if path:
        print(f'Load model: {path}')
        checkpoint = torch.load(path)
        m.load_state_dict(checkpoint['model'])

    return m


def sampling(model: clm_rnn.NNLM, tokens: list, n: int, limit=cfg.MAX_SEQ_LEN) -> list:

    def token2tensor(t):
        tensor = torch.zeros(len(tokens), dtype=torch.float).to(cfg.DEVICE)
        tensor[tokens.index(t)] = 1.
        return tensor

    model.eval()
    smi_list = []
    for i in range(n):
        # initialize
        predict_token, end_token = token2tensor(cfg.BOS), token2tensor(cfg.EOS)
        hidden = model.init_states(1)
        smiles = ''
        # generating
        while (not predict_token.equal(end_token)) and len(smiles) < limit:
            output, hidden = model(predict_token.reshape(1, 1, -1), hidden)
            output = F.softmax(output)
            output = output.data.reshape(-1)
            idx = torch.multinomial(output, 1).int().item()
            token = tokens[idx]
            smiles += token
            predict_token = token2tensor(token)
        # print(len(smiles), smiles)
        smi_list.append(smiles)

    return smi_list


def generate(n: int, clm: clm_rnn.NNLM, smiles: list, tokens: list) -> tuple:
    # sampling
    print('Sampling ...')
    generate_smiles = sampling(clm, tokens, n=n)
    print('Sampling Finished !')
    print('Checking ...')
    valid_smiles = [smi_tools.to_canonical_smi(smi) for smi in generate_smiles]
    valid_smiles = [smi for smi in valid_smiles if smi is not None]
    unique_smiles = list(set(valid_smiles))
    novel_smiles = [smi for smi in unique_smiles if smi not in smiles]
    print('Checking Finished !')
    record = [n, len(valid_smiles), len(unique_smiles), len(novel_smiles)]
    return novel_smiles, record


def normed_predict(x: torch.tensor, m: reg_rnn.RNNPredictor, norm: tools.ZeroSoreNorm) -> torch.tensor:

    m.eval()
    y_predict = reg_rnn.predict(m, x)
    y_predict = y_predict.squeeze().detach().cpu().numpy()
    y_predict = norm.recovery(y_predict)
    return y_predict


def smiles_predict(smiles: list, tokens: list, reg: reg_rnn.RNNPredictor, norm: tools.ZeroSoreNorm) -> torch.tensor:

    smiles_ori = [smi for smi in smiles if smi_tools.if_oov_exclude(smi, tokens)]
    smiles = [smi for smi in smiles_ori]
    oh = [smi_tools.smiles2tensor(smi, tokens) for smi in smiles]
    y_predict = normed_predict(oh, reg, norm)
    return y_predict