import logging
from abc import abstractmethod


class MetaMethod:
    output_list = {}  # 原始关键词:预测关键词    str:str
    stopwords = set()  # 停用词列表
    test_documents = []  # 预处理后的测试集文档列表
    train_documents = []  # 预处理后的训练集文档列表
    cost: int  # 提取关键词时间花费

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
        pass

    @abstractmethod
    def computer_metric(self):
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
