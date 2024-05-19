import logging
from typing import Union

import jieba.posseg as pseg
import numpy as np
from pandas.core.arrays import ExtensionArray
from textrank4zh import TextRank4Keyword


class TEXTRANK:
    stopwords = set()
    documents = []
    top_keywords_per_document = []
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
            document = []
            words = pseg.cut(text)
            for word, flag in words:
                if flag.startswith('n') and word not in self.stopwords and len(word) > 1:
                    document.append(word)
            self.documents.append(document)

    def keyword_extraction(self):
        logging.info("MODEL: " + "TEXTRANK, " + "OPERATION: " + "load_stopwords")
        self.load_stopwords()
        logging.info("MODEL: " + "TEXTRANK, " + "OPERATION: " + "filter_documents")
        self.filter_documents()
        tr4w = TextRank4Keyword()

        for document in self.documents:
            text = ' '.join(document)
            tr4w.analyze(text=text, lower=True, window=2)
            keywords = tr4w.get_keywords(3, word_min_len=2)
            self.top_keywords_per_document.append([keyword.word for keyword in keywords])

        self.build_output_list()

    def build_output_list(self):
        for keywords in enumerate(self.top_keywords_per_document):
            self.output_list[self.inputs[1].tolist()[keywords[0]]] = keywords[1]

# if __name__ == '__main__':
#     stopwords_file = r"D:\Program Files\GithubRepositorySet\NLP\cfsfdp_algo\data\stop_words\cn_stopwords.txt"
#     textrank = TEXTRANK(stopwords_file=stopwords_file, inputs=read_test_tsv())
#     textrank.keyword_extraction()
#     print(textrank.output_list)
