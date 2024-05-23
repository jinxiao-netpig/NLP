from abc import abstractmethod


class MetaMethod:
    output_list = {}  # 原始关键词:预测关键词

    @abstractmethod
    def __init__(self):
        pass

    def load_stopwords(self):
        pass

    def filter_documents(self):
        pass

    @abstractmethod
    def keyword_extraction(self):
        pass
