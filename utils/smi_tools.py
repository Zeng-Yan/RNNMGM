import re
import torch
import copy
import numpy as np
from rdkit import Chem
import configs as cfg


def to_canonical_smi(smi: str):
    if cfg.BOS in smi:  # 去掉起始符
        smi = smi.replace(cfg.BOS, '')
    try:
        mol = Chem.MolFromSmiles(smi)
        Chem.SanitizeMol(mol)
        canonical_smi = Chem.MolToSmiles(mol, canonical=True)
        return canonical_smi
    except:
        # print('[WARNING] Invalid SMILES found: {}'.format(smi))
        return None


def tokenize(smiles: str) -> list:
    """
    tokenize a SMILES and return a list containing all the tokens after tokenization.
    eg：
        O=[N+]([O-])
        -->
        ['O', '=', '[N+]', '(', '[O-]', ')']
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

    tokens = split_by(smiles, REGEXP_ORDER)

    return tokens


def gather_tokens(smiles: list, single_split=False) -> list:

    tokenized_smiles = [[c for c in smi] for smi in smiles] if single_split else [tokenize(smi) for smi in smiles]
    token_set = set()
    for tokenized_smi in tokenized_smiles:
        token_set.update(tokenized_smi)
    token_set = sorted(token_set)
    return list(token_set)


def if_oov_exclude(smiles: str, tokens: list, single_split=False) -> bool:

    tks = [c for c in smiles] if single_split else tokenize(smiles)
    differ = [token for token in tks if token not in tokens]
    return False if differ else True


def if_each_fold_cover_all_tokens(folds: list, tokens: list) -> bool:

    for i, fold in enumerate(folds):
        train_smi = fold[0]
        fold_tokens = gather_tokens([cfg.BOS + smi + cfg.EOS for smi in train_smi], single_split=cfg.SINGLE_TOKENIZE)
        print(f'Fold {i}, Tokens: {fold_tokens}')
        if fold_tokens != tokens:
            return False
    return True


def tokenize_smiles(smiles: str, tokens: list) -> list:

    frags = []
    while smiles:
        snapshot = smiles
        for token in tokens:
            length = len(token)
            if smiles[0:length] == token:
                # 第二位检测，避免cl被c替换
                sub = smiles[0:length+1]
                sub = sub if sub in ['Cl', 'Br', 'Si'] else token
                smiles = smiles.replace(sub, '', 1)
                frags.append(sub)

        if snapshot == smiles:
            raise RuntimeError(f'A substring \'{smiles}\' includes token which not in the token list {tokens}!')

    return frags


def smiles2tensor(smiles: str, tokens: list, encoding=True) -> torch.tensor:
    """
    convert a SMILES into an One-Hot tensor if encoding is True, else a label tensor.
    :param smiles:
    :param tokens:
    :param encoding:
    :return:
    """

    tokenized_smiles = tokenize_smiles(smiles, tokens)
    labeled_smiles = torch.tensor([tokens.index(x) for x in tokenized_smiles])

    if encoding:
        oh_tensor = torch.zeros(len(labeled_smiles), len(tokens))
        oh_tensor[range(len(oh_tensor)), labeled_smiles] = 1
        return oh_tensor
    else:
        return labeled_smiles


def smiles2ecfp(smi_list: list, bits=2048) -> np.array:

    from rdkit.Chem import AllChem

    all_fp = np.zeros((len(smi_list), bits))
    for idx, smi in enumerate(smi_list):
        mol = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        all_fp[idx] = fp
    return all_fp


def get_sa(smiles: list) -> list:
    from rdkit.Chem import AllChem as Chem
    from utils.SA_Score import sascorer

    lst_mol = [Chem.MolFromSmiles(smi) for smi in smiles]
    sa = [sascorer.calculateScore(mole) if mole else float('inf') for mole in lst_mol]
    return sa


def sum_over_bonds(mol_list, predefined_bond_types=[], return_names=True):
    """
    copy from https://github.com/delton137/mmltoolkit
    """
    from collections import defaultdict

    if (isinstance(mol_list, list) == False):
        mol_list = [mol_list]

    empty_bond_dict = defaultdict(lambda: 0)
    num_mols = len(mol_list)

    if (len(predefined_bond_types) == 0):
        # first pass through to enumerate all bond types in all molecules and set them equal to zero in the dict
        for i, mol in enumerate(mol_list):
            bonds = mol.GetBonds()
            for bond in bonds:
                bond_start_atom = bond.GetBeginAtom().GetSymbol()
                bond_end_atom = bond.GetEndAtom().GetSymbol()
                bond_type = bond.GetSmarts(allBondsExplicit=True)
                bond_atoms = [bond_start_atom, bond_end_atom]
                if (bond_type == ''):
                    bond_type = "-"
                bond_string = min(bond_atoms) + bond_type + max(bond_atoms)
                empty_bond_dict[bond_string] = 0
    else:
        for bond_string in predefined_bond_types:
            empty_bond_dict[bond_string] = 0

    # second pass through to construct X
    bond_types = list(empty_bond_dict.keys())
    num_bond_types = len(bond_types)

    X_LBoB = np.zeros([num_mols, num_bond_types])

    for i, mol in enumerate(mol_list):
        bonds = mol.GetBonds()
        bond_dict = copy.deepcopy(empty_bond_dict)
        for bond in bonds:
            bond_start_atom = bond.GetBeginAtom().GetSymbol()
            bond_end_atom = bond.GetEndAtom().GetSymbol()
            # skip dummy atoms
            if (bond_start_atom == '*' or bond_end_atom == '*'):
                pass
            else:
                bond_type = bond.GetSmarts(allBondsExplicit=True)
                if (bond_type == ''):
                    bond_type = "-"
                bond_atoms = [bond_start_atom, bond_end_atom]
                bond_string = min(bond_atoms) + bond_type + max(bond_atoms)
                bond_dict[bond_string] += 1

        # at the end, pick out only the relevant ones
        X_LBoB[i, :] = [bond_dict[bond_type] for bond_type in bond_types]

    if (return_names):
        return bond_types, X_LBoB
    else:
        return X_LBoB


if __name__ == '__main__':
    s = ['NN([N]c1nonc1[N][N]c1nonc1[N]N([O])c1nonc1N[N+](=O)[O-])c1nonc1N[N+](=O)[O-]',
         '[N]N([N]c1nonc1[N][N]c1nonc1[N]N(N)[N+](=O)[O-])c1nonc1N[N+](=O)[O-]',
         'O=C1c2[c][c]c([N+](=O)[O-])[c]c2[C]2[C][C]=[C][N]N21']

    tokens = gather_tokens(s)
    print(tokens)
    oh = smiles2tensor('O=C1c2[c][c]c([N+](=O)[O-])[c]c2[C]2[C][C]=[C][N]N21', tokens)
    print(oh)
