import os
import logging
from datetime import datetime
import re

class Logger:
    ####################################################
    # logger = Logger()
    # logger.log_message("This is a logger message.")
    ####################################################
    def __init__(self, log_dir='./Logs', log_name=None):
        if log_name is None:
            now = datetime.now()                                  # 获取当前日期和时间
            log_name = now.strftime("%Y-%m-%d_%H-%M-%S.log")  # 将日期和时间格式化为字符串，作为日志文件的名称

        # 创建一个记录器
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # 创建一个处理器，将日志写入文件
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        file_handler = logging.FileHandler(os.path.join(log_dir, log_name))
        file_handler.setLevel(logging.DEBUG)

        # 创建一个处理器，将日志打印到控制台
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        # 创建一个日志格式
        formatter = logging.Formatter('[%(asctime)s] - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 将处理器添加到记录器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def log_message(self, message):
        # 使用记录器记录信息
        self.logger.info(message)

# 数据类型转换函数
def convert_value(value):
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value

# 需保留的参数
model_args = {
    # log_name: saved_name
    'rating_reg': 'rating_reg',
    # 'context_reg': 'context_reg',
    # 'text_reg': 'text_reg',
    'Model:': 'model',
    'emsize': 'emsize',
}
rate_args = {
    'RMSE': 'RMSE',
    'MAE': 'MAE',
}
text_args = {
    'BLEU-1': 'BLEU-1',
    'BLEU-4': 'BLEU-4',
    'DIV': 'DIV',
    'FCR': 'FCR',
    # 'USR': 'USR',
    'FMR': 'FMR',
    'rouge_1/f_score': 'R1-F',
    'rouge_1/r_score': 'R1-R',
    'rouge_1/p_score': 'R1-P',
    'rouge_2/f_score': 'R2-F',
    'rouge_2/r_score': 'R2-R',
    'rouge_2/p_score': 'R2-P',
}

class LogReader:
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

    def read_log(self, file_path, rate=True, text=True):
        pattern = re.compile(r' - (\S+)\s+(.+)')  # 正则表达式

        args = {}
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                log_text = file.read()

                # 查找所有匹配项
                matches = {key: val for key, val in pattern.findall(log_text)}

                for key, val in self.kwargs.items():  # 自定义项目
                    if key in matches.keys():
                        args[val] = convert_value(matches[key])
                    else:
                        args[val] = None

                if 'index_dir' in matches.keys():
                    args['dataset'] = convert_value(matches['index_dir'][8:-3])
                    args['data_id'] = convert_value(matches['index_dir'][-2])

                for key, val in model_args.items():  # model 相关项
                    if key in matches.keys():
                        args[val] = convert_value(matches[key])
                    else:
                        args[val] = None

                if rate:  # 评分预测相关指标
                    for key, val in rate_args.items():
                        if key in matches.keys():
                            args[val] = convert_value(matches[key])
                        else:
                            args[val] = None

                if text:  # 文本预测相关指标
                    if 'USR' in matches.keys():
                        args['USR'] = convert_value(matches['USR'][:6])
                        # args['USN'] = convert_value(matches['USR'][12:].strip())
                    for key, val in text_args.items():
                        if key in matches.keys():
                            args[val] = convert_value(matches[key])
                        else:
                            args[val] = None
        return args

    def read_logs(self, log_dir, rate=True, text=True):
        args_list = []
        for filename in os.listdir(log_dir):
            file_path = os.path.join(log_dir, filename)
            args_list.append(self.read_log(file_path, rate, text))
        return args_list