import re
import torch
import copy
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from collections import defaultdict

import configs


def split_smi(smi: str) -> list:
    """
    输入一个字符串，返回分词后的token列表
    eg：
        O=[N+]([O-])
        -->
        ['O', '=', '[N+]', '(', '[O-]', ')']

    :param smi: smiles字符串；str
    :return: list(str)
    """

    REGEXPS = {
        "brackets": re.compile(r"(\[[^\]]*\])"),
        "2_ring_nums": re.compile(r"(%\d{2})"),
        "brcl": re.compile(r"(Br|Cl|Si)")
    }
    REGEXP_ORDER = ["brackets", "2_ring_nums", "brcl"]

    def split_by(smi, regexps):
        if not regexps:
            return list(smi)
        regexp = REGEXPS[regexps[0]]
        splitted = regexp.split(smi)
        tokens = []
        for i, split in enumerate(splitted):
            if i % 2 == 0:
                tokens += split_by(split, regexps[1:])
            else:
                tokens.append(split)
        return tokens

    tokens = split_by(smi, REGEXP_ORDER)

    return tokens


def get_list_of_tokens(list_of_smi: list, single_split=False) -> list:
    """
    统计smiles列表中的token集合，返回token集合列表
    :param list_of_smi: smiles字符串列表；list(str)
    :param single_split: 是否按单个字符切分smiles字符串；bool
    :return: list(str)
    """

    if single_split:
        list_of_tokenized_smi = [[c for c in x] for x in list_of_smi]
        set_of_token = set()
        for tokenized_smi in list_of_tokenized_smi:
            set_of_token.update(tokenized_smi)

        set_of_token = sorted(set_of_token)

        return list(set_of_token)

    else:
        list_of_tokenized_smi = [split_smi(x) for x in list_of_smi]
        set_of_token = set()
        for tokenized_smi in list_of_tokenized_smi:
            set_of_token.update(tokenized_smi)

        set_of_token = sorted(set_of_token)

        return list(set_of_token)


def smi2tokens(smi: str, list_of_tokens: list) -> list:
    """
    根据token集合列表划分一个smiles字符串为片段列表
    :param smi: smiles字符串；str
    :param list_of_tokens:
    :return: list(str)
    """

    frags = []
    while smi:
        old_s = smi
        for token in list_of_tokens:
            length = len(token)
            if smi[0:length] == token:
                if smi[0:length+1] == 'Cl':
                    smi = smi.replace('Cl', '', 1)
                    frags.append('Cl')
                elif smi[0:length+1] == 'Br':
                    smi = smi.replace('Br', '', 1)
                    frags.append('Br')
                elif smi[0:length+1] == 'Si':
                    smi = smi.replace('Si', '', 1)
                    frags.append('Si')
                else:
                    smi = smi.replace(token, '', 1)
                    frags.append(token)

        if old_s == smi:
            raise RuntimeError('A substring \'{}\' includes token which not in the token list {}!'
                               .format(smi, list_of_tokens))

    return frags


def smi2oh(smi: str, list_of_tokens: list) -> torch.tensor:
    """
    将一个smiles字符串转为one-hot编码后的tensor

    :param smi: smiles字符串；str
    :param list_of_tokens: list(str)
    :return: torch.tensor
    """

    smi_tokenized = smi2tokens(smi, list_of_tokens)
    smi_labeled = torch.tensor([list_of_tokens.index(x) for x in smi_tokenized])

    smi_oh = torch.zeros(len(smi_labeled), len(list_of_tokens))
    smi_oh[range(len(smi_oh)), smi_labeled] = 1

    return smi_oh


def smi_list2oh_mat(smi_list: list, list_of_tokens: list, max_len: int) -> torch.tensor:
    """
    将smiles字符串列表转为one-hot编码后的tensor

    :param smi_list: smiles字符串列表；list(str)
    :param list_of_tokens: list(str)
    :param max_len: smiles字符串的最大长度；int
    :return: torch.tensor
    """

    list_of_tokenized_smi = [smi2tokens(smi, list_of_tokens) for smi in smi_list]
    mat_of_labeled_smi = torch.zeros(len(smi_list), max_len)  # [n, max_len]

    for i in range(len(list_of_tokenized_smi)):
        for j in range(len(list_of_tokenized_smi[i])):
            mat_of_labeled_smi[i, j] = list_of_tokens.index(list_of_tokenized_smi[i][j])

    mat_of_ohe = torch.zeros(len(smi_list), max_len, len(list_of_tokens))

    for i in range(len(list_of_tokenized_smi)):
        for j in range(len(list_of_tokenized_smi[i])):
            idx = int(mat_of_labeled_smi[i, j].item())
            mat_of_ohe[i, j, idx] = 1

    return mat_of_ohe, mat_of_labeled_smi


def smi2fp(smi_list: list, bits=2048) -> np.array:
    """
    将smiles字符串列表转为分子指纹矩阵

    :param smi_list: smiles字符串列表；list(str)
    :param bits: ECFP指纹编码位数；int
    :return: numpy.array
    """

    all_fp = np.zeros((len(smi_list), bits))
    for idx, smi in enumerate(smi_list):
        mol = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        all_fp[idx] = fp
    return all_fp


def to_canonical_smi(smi: str):
    if configs.BOS in smi:  # 去掉起始符
        smi = smi.replace(configs.BOS, '')
    try:
        mol = Chem.MolFromSmiles(smi)
        Chem.SanitizeMol(mol)
        canonical_smi = Chem.MolToSmiles(mol, canonical=True)
        return canonical_smi
    except:
        # print('wrong smiles: {}'.format(smi))
        return None

