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
    while center_indice_count > 4 and epoch < 2000:
        epoch += 1
        epsilon *= (1 + learn_rate)
        threshold *= (1 + learn_rate)
        cfsfdp_instance.set_epsilon(epsilon=epsilon)
        cfsfdp_instance.set_threshold(threshold=threshold)
        logging.info("epsilon: ", cfsfdp_instance.epsilon)
        logging.info("threshold: ", cfsfdp_instance.threshold)
        logging.info("center_indice_count: ", len(cfsfdp_instance.center_indices_list))
        logging.info("epoch: ", epoch)
        cfsfdp_instance.fit()
        
        center_indice_count = len(cfsfdp_instance.center_indices_list)

    logging.info("epsilon: ", epsilon)
    logging.info("threshold: ", threshold)
    write_config((epsilon, threshold))
