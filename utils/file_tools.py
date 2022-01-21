import csv


def load_data_from_csv(csv_path: str, with_head=False):
    """
    读取csv文件，默认csv中无标题

    :param csv_path:
    :param with_head: 是否带有标题行
    :return:
    """
    with open(csv_path, 'r') as f:
        data = csv.reader(f)
        data = list(data)

    if with_head:
        return data[1:], data[0]
    else:
        return data, None


def save_data_to_csv(csv_path: str, data: list, head: list):
    """
    将列表数据写入到csv文件中
    :param csv_path:
    :param data:
    :param head:
    :return:
    """

    with open(csv_path, 'w+', newline='') as f:
        f_csv = csv.writer(f)
        if head:
            f_csv.writerow(head)
        f_csv.writerows(data)

    return 0

