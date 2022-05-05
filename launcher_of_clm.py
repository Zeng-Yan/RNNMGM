import torch
import torch.optim as optim

import configs as cfg
from utils import builder, smi_tools, tools
from models import clm_rnn


def train_clm(data_path: str, smi_idx: int, model_name='clm', epochs=100, fq_saving=5,
              model_path=None, pre_data_path=None, pre_smi_idx=None):

    # processing smiles
    data, _ = tools.load_data_from_csv(data_path, with_head=True)  # 从文件读入数据
    smiles = [cfg.BOS + x[smi_idx] + cfg.EOS for x in data]  # 获取smiles
    tokens = smi_tools.gather_tokens(smiles, single_split=cfg.SINGLE_TOKENIZE)  # 获取字符集合列表
    print(f'Tokens: {tokens}')

    #
    if pre_data_path:
        data, _ = tools.load_data_from_csv(pre_data_path, with_head=True)  # 从文件读入数据
        pre_smiles = [cfg.BOS + x[pre_smi_idx] + cfg.EOS for x in data]  # 获取smiles
        tokens = smi_tools.gather_tokens(pre_smiles, single_split=cfg.SINGLE_TOKENIZE)  # 获取字符集合列表
        print(f'Tokens of pre-trained data: {tokens}')
        print(f'There are {len(smiles)} SMILES strings in data.')
        smiles = [smi for smi in smiles if smi_tools.if_oov_exclude(smi, tokens, single_split=cfg.SINGLE_TOKENIZE)]
        print(f'There are {len(smiles)} SMILES strings after checking tokens.')

    loader = builder.clm_packer(smiles, tokens)  # 封装数据

    # initialize clm
    m = builder.build_clm(len(tokens), model_path)
    # initial optimizer
    opt = optim.Adam(m.parameters(), lr=cfg.CLM_LR_RATE)

    # training
    records = clm_rnn.train(model=m, optimizer=opt, data_loader=loader,
                            epochs=epochs, fq_of_save=fq_saving, name=model_name)

    return 0


def generate(n: int, idx: int, data_path: str, model_path: str, saving_path: str) -> list:
    # processing smiles
    data, head = tools.load_data_from_csv(data_path, with_head=True)  # 从文件读入数据
    smiles = [x[idx] for x in data]  # 获取smiles
    raw_smiles = [smi_tools.to_canonical_smi(smi) for smi in smiles]
    raw_smiles = [smi for smi in raw_smiles if smi is not None]

    # initialize clm
    tokens = smi_tools.gather_tokens([cfg.BOS + smi + cfg.EOS for smi in smiles], single_split=cfg.SINGLE_TOKENIZE)  # 获取字符集合列表
    m = builder.build_clm(len(tokens), model_path)

    # sampling
    print('Sampling ...')
    novel_smiles, record = builder.generate(n, m, raw_smiles, tokens)
    print('Sampling Finished !')
    print(f'Sample:{n}, Valid:{len(record[1])}, Unique:{len(record[2])}, Novel:{len(record[3])}')
    tools.save_data_to_csv(saving_path, [[smi] for smi in novel_smiles], ['smiles'])

    return novel_smiles


def valid_generate(valid: int, idx: int, data_path: str, model_path: str, saving_path: str, tokens: list) -> list:
    # processing smiles
    data, head = tools.load_data_from_csv(data_path, with_head=True)  # 从文件读入数据
    smiles = [x[idx] for x in data]  # 获取smiles
    raw_smiles = [smi_tools.to_canonical_smi(smi) for smi in smiles]
    raw_smiles = [smi for smi in raw_smiles if smi is not None]

    # initialize clm
    if not tokens:
        tokens = smi_tools.gather_tokens([cfg.BOS + smi + cfg.EOS for smi in smiles], single_split=cfg.SINGLE_TOKENIZE)  # 获取字符集合列表
    m = builder.build_clm(len(tokens), model_path)

    # sampling
    valid_smiles = []
    while valid > 0:
        generate_smiles = builder.sampling(m, tokens, n=1)
        generate_smiles = smi_tools.to_canonical_smi(generate_smiles[0])
        if generate_smiles:
            valid -= 1
            valid_smiles.append(generate_smiles)

    unique_smiles = list(set(valid_smiles))
    novel_smiles = [smi for smi in unique_smiles if smi not in raw_smiles]
    tools.save_data_to_csv(saving_path, [[smi] for smi in valid_smiles], ['smiles'])
    print(f'Valid: {len(valid_smiles)}, Unique: {len(unique_smiles)}, Novel: {len(novel_smiles)}')

    return valid_smiles


if __name__ == '__main__':
    # 预训练
    train_clm(data_path='data/Ds_5.csv', smi_idx=0, model_name='pt', epochs=30, fq_saving=5)
    