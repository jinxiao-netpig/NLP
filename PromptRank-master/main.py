import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import logging
import time
import torch

from data import data_process
from inference import keyphrases_selection
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration


def get_setting_dict():
    """
    获取该关键词模型的一些配置参数
    """
    setting_dict = {}
    setting_dict["max_len"] = 512
    setting_dict["temp_en"] = "Book:"
    setting_dict["temp_de"] = "This book mainly talks about "
    setting_dict["model"] = "base"
    setting_dict["enable_filter"] = False
    setting_dict["enable_pos"] = True
    setting_dict["position_factor"] = 1.2e8
    setting_dict["length_factor"] = 0.6
    return setting_dict


def parse_argument():
    """
    解析命令行参数
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input dataset.")
    parser.add_argument("--dataset_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The input dataset name.")
    parser.add_argument("--batch_size",
                        default=None,
                        type=int,
                        required=True,
                        help="Batch size for testing.")
    parser.add_argument("--log_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="Path for Logging file")
    args = parser.parse_args()
    return args


def main():
    # 获得关键词模型的配置参数
    setting_dict = get_setting_dict()
    # 获得命令行参数
    args = parse_argument()

    # 获得设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 初始化日志服务
    log = Logger(args.log_dir + args.dataset_name + '.log')
    # 记录模型启动时间
    start = time.time()
    log.logger.info("Start Testing ...")

    # 处理数据
    dataset, doc_list, labels, labels_stemed = data_process(setting_dict, args.dataset_dir, args.dataset_name)
    dataloader = DataLoader(dataset, num_workers=4, batch_size=args.batch_size)
    # 实例化T5模型
    model = T5ForConditionalGeneration.from_pretrained(
        r"C:\Users\HP\.cache\huggingface\hub\models--t5-base\snapshots\a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1")
    # 配置模型的设备
    model.to(device)

    # 关键词提取具体实现
    keyphrases_selection(setting_dict, doc_list, labels_stemed, labels, model, dataloader, device, log)

    end = time.time()
    log_setting(log, setting_dict)
    log.logger.info("Processing time: {}".format(end - start))


def log_setting(log, setting_dict):
    for i, j in setting_dict.items():
        log.logger.info(i + ": {}".format(j))


class Logger(object):

    def __init__(self, filename, level='info'):
        level = logging.INFO if level == 'info' else logging.DEBUG
        self.logger = logging.getLogger(filename)
        self.logger.propagate = False
        # # format_str = logging.Formatter(fmt)  
        # if args.local_rank == 0 :
        #     level = level
        # else:
        #     level = 'warning'
        self.logger.setLevel(level)  # 

        th = logging.FileHandler(filename, 'w')
        # formatter = logging.Formatter('%(asctime)s => %(name)s * %(levelname)s : %(message)s')
        # th.setFormatter(formatter)

        # self.logger.addHandler(sh)  #
        self.logger.addHandler(th)  # 


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    main()
