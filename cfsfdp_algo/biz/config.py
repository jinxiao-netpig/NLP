import logging

import yaml


def read_config() -> tuple[float, float, float]:
    logging.info("read config from config.yaml")
    with open('../configs/config.yaml', 'r', encoding='utf8') as file:
        configs = yaml.safe_load(file)
    return configs['cfsfdp']['epsilon'], configs['cfsfdp']['threshold'], configs['cfsfdp']['learn_rate']


def read_train_config() -> tuple[float, float, float]:
    logging.info("read config from config_train.yaml")
    with open('../configs/config_train.yaml', 'r', encoding='utf8') as file:
        configs = yaml.safe_load(file)

    return configs['cfsfdp']['epsilon'], configs['cfsfdp']['threshold'], configs['cfsfdp']['learn_rate']


def write_config(data: tuple[float, float]):
    logging.info("write to config.yaml")
    try:
        with open('../configs/config.yaml', 'w', encoding='utf8') as file:
            write_data = {}
            write_data['cfsfdp']['epsilon'] = data[0]
            write_data['cfsfdp']['threshold'] = data[1]
            yaml.dump(write_data, file)
    except Exception as e:
        logging.error("write error: ", e)
