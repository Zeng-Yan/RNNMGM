from rdkit import Chem
from rdkit.Chem import AllChem as Chem


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


