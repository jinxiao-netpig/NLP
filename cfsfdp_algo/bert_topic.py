import logging
from typing import Union

import numpy as np
from bertopic import BERTopic
from pandas.core.arrays import ExtensionArray

from biz.data import read_test_tsv


class BERTOPIC:
    stopwords = set()
    documents = []
    output_list = {}  # 原始关键词:预测关键词

    def __init__(self, stopwords_file: str,
                 inputs: tuple[Union[ExtensionArray, np.ndarray], Union[ExtensionArray, np.ndarray]]):
        self.stopwords_file = stopwords_file
        self.inputs = inputs
        pass

    def load_stopwords(self):
        with open(self.stopwords_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.stopwords.add(line.strip())

    def filter_documents(self):
        for text in self.inputs[0].tolist():
            document = [text]
            self.documents.append(document)

    def keyword_extraction(self):
        logging.info("MODEL: " + "BERTOPIC, " + "OPERATION: " + "load_stopwords")
        self.load_stopwords()
        logging.info("MODEL: " + "BERTOPIC, " + "OPERATION: " + "filter_documents")
        self.filter_documents()
        topic_model = BERTopic(language="multilingual")
        y = []
        for keywords in enumerate(self.inputs[1].tolist()):
            y.append(keywords[0])
        topics, probs = topic_model.fit_transform(documents=self.documents[0], y=y)
        self.result = topic_model.get_document_info(self.documents)


if __name__ == '__main__':
    stopwords_file = r"D:\Program Files\GithubRepositorySet\NLP\cfsfdp_algo\data\stop_words\cn_stopwords.txt"
    bertopic = BERTOPIC(stopwords_file=stopwords_file, inputs=read_test_tsv())
    bertopic.keyword_extraction()
    print(bertopic.result)
