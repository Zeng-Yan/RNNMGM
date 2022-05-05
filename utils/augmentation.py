from rdkit import Chem
from rdkit.Chem import Recap
from rdkit.Chem import AllChem as Chem
import random


def get_rand_smi(smi, max_folds=10):
    # 由一个smiles得到其多个不同写法，输入字符串，返回列表

    mol = Chem.MolFromSmiles(smi)
    smi_list = []
    for _ in range(max_folds):
        try:
            s = Chem.MolToSmiles(mol, doRandom=True)
            smi_list.append(s)
        except BaseException:
            continue

    smi_list = set(smi_list)
    max_folds = min(max_folds, len(smi_list))

    smi_list = smi_list - {smi}
    smi_list = list(smi_list)[0:max_folds - 1] + [smi]

    return smi_list


def augmentation_by_smi(lst_smi: list, max_folds=10):
    """
    对一个列表中的smiles做数据增强，返回增强后的列表
    """

    if max_folds <= 1:
        return lst_smi

    list_of_augmented_smi = []
    for idx in range(len(lst_smi)):
        smi = lst_smi[idx]
        list_of_rand_smi = get_rand_smi(smi, max_folds)
        list_of_augmented_smi += list_of_rand_smi

    return list_of_augmented_smi


def augmentation_by_enum(lst_smi, props, max_times=10):
    if max_times <= 1:
        return lst_smi

    augmented_smi, augmented_p = [], []
    for idx in range(len(lst_smi)):
        smi = lst_smi[idx]
        prop = props[idx]

        lst_rand_smi = get_rand_smi(smi, max_times)
        lst_corr_p = [prop for s in lst_rand_smi]

        augmented_smi += lst_rand_smi
        augmented_p += lst_corr_p

    return augmented_smi, augmented_p


def replace_number(smi: str, n=1) -> str:
    s = ''
    for char in smi:
        new_char = str(int(char) + n) if char.isdigit() else char
        s += new_char
    return s


def combine_fragments(fragments: list, n_combs: int):
    # print('start molecular fragments combination.')
    list_of_smi = []
    singles = [frag for frag in fragments if frag.count('*') == 1]
    n_frags, n_singles = len(fragments), len(singles)
    print(f'{n_frags} {n_singles}')
    for idx in [random.randint(0, n_frags - 1) for _ in range(n_combs)]:
        bone = fragments[idx]
        bone = replace_number(bone)
        print('\n', bone)
        for i in range(bone.count('*')):
            frag = singles[random.randint(0, n_singles - 1)]
            print(frag)
            frag = frag.replace('*', '')
            bone = bone.replace('*', frag, 1)
        bone = bone.replace('*', '')
        print(bone)
        try:  # 验证组合是否正确
            mol = Chem.MolFromSmiles(bone)
            Chem.SanitizeMol(mol)
            canonical_smi = Chem.MolToSmiles(mol, canonical=True)
            list_of_smi.append(canonical_smi)
            print('Valid')
        except:
            print('Failed')
            pass
    return list_of_smi


def augmentation_by_fragment(list_of_smi: list, n: int):
    """

    :param list_of_smi:
    :param n:
    :return:
    """

    fragments = set()
    for idx, smi in enumerate(list_of_smi):
        m = Chem.MolFromSmiles(smi)
        print(idx, smi)

        hierarch = Recap.RecapDecompose(m)
        leaves = list(hierarch.GetLeaves().keys())
        print(leaves)

        fragments.update(leaves)

    fragments = list(fragments)
    print('get {} fragments'.format(len(fragments)))

    return combine_fragments(fragments, n)
