import logging
from abc import abstractmethod


class MetaMethod:
    output_list: dict[str, list[str]] = {}  # 原始关键词:预测关键词    str:list[str]
    stopwords = set()  # 停用词列表
    test_documents = []  # 预处理后的测试集文档列表
    train_documents = []  # 预处理后的训练集文档列表
    cost: int  # 提取关键词时间花费
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    precision: float
    recall: float
    f_score: float

    @abstractmethod
    def __init__(self):
        pass

    def load_stopwords(self):
        logging.info("MODEL: " + str(type(self)) + ", " + "OPERATION: " + "load_stopwords")
        pass

    def filter_documents(self, dataset_name: str):
        logging.info("MODEL: " + str(type(self)) + ", " + "OPERATION: " + "filter_documents")
        pass

    @abstractmethod
    def keyword_extraction(self, dataset_name: str):
        self.load_stopwords()
        self.filter_documents(dataset_name)
        pass

    def compute_metric(self):
        """
        计算 P、R、F1

        :return:
        """

        tp_fn_sum = 0
        for oris, preds in self.output_list.items():
            for pred in preds:
                if pred in oris:
                    self.tp += 1
                else:
                    self.fp += 1
            oris_list = oris.split(sep=";")
            tp_fn_sum += len(oris_list)
        self.fn = tp_fn_sum - self.tp

        self.precision = self.tp / (self.tp + self.fp)
        self.recall = self.tp / (self.tp + self.fn)
        self.f_score = 2 * self.precision * self.recall / (self.recall + self.precision)

    def train_model(self):
        # 目前训练出来的模型参数写死在函数体
        pass

    @abstractmethod
    def download_data(self):
        pass

    def show_output_list(self):
        for k in self.output_list.keys():
            print("origin: " + k)
            print("pred: " + ";".join(self.output_list[k]))
        logging.info("output_list length: " + str(len(self.output_list)))

    def show_keywords_extraction_cost(self):
        print("cost: " + str(self.cost))
