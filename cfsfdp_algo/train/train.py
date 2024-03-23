import logging

from biz.config import read_train_config, write_config
from biz.data import read_data
from cfsfdp import CFSFDP


def train():
    points = read_data()
    epsilon, threshold, learn_rate = read_train_config()
    cfsfdp_instance = CFSFDP(epsilon=epsilon, threshold=threshold, points=points)
    cfsfdp_instance.fit()
    center_indice_count = len(cfsfdp_instance.center_indices_list)
    epoch = 0

    # 迭代模型，调整参数
    # todo: 有问题，这里每次都要创建新的实例，需要修改
    while center_indice_count > 4 and epoch < 2000:
        epoch += 1
        epsilon *= (1 + learn_rate)
        threshold *= (1 + learn_rate)
        cfsfdp_instance = CFSFDP(epsilon=epsilon, threshold=threshold, points=points)
        cfsfdp_instance.fit()
        center_indice_count = len(cfsfdp_instance.center_indices_list)

    logging.info("epsilon: ", epsilon)
    logging.info("threshold: ", threshold)
    write_config((epsilon, threshold))