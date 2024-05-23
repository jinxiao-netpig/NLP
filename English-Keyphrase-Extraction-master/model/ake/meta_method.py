import logging
from abc import abstractmethod


class MetaMethod:
    output_list = {}  # 原始关键词:预测关键词
    stopwords = set()  # 停用词列表

    @abstractmethod
    def __init__(self):
        pass

    def load_stopwords(self):
        logging.info("MODEL: " + str(type(self)) + ", " + "OPERATION: " + "load_stopwords")
        pass

    def filter_documents(self):
        logging.info("MODEL: " + str(type(self)) + ", " + "OPERATION: " + "filter_documents")
        pass

    @abstractmethod
    def keyword_extraction(self):
        pass

    @abstractmethod
    def computer_metric(self):
        pass

    @abstractmethod
    def download_data(self):
        pass
