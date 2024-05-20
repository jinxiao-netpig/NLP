import logging
from typing import Union

import jieba.posseg as pseg
import kex
import numpy as np
from pandas.core.arrays import ExtensionArray


class SINGLERANK:
    output_list = {}  # 原始关键词:预测关键词
    documents = []
    raw_data = []
    stopwords = set()

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
        logging.info("MODEL: " + "SINGLERANK, " + "OPERATION: " + "load_stopwords")
        self.load_stopwords()
        logging.info("MODEL: " + "SINGLERANK, " + "OPERATION: " + "filter_documents")
        self.filter_documents()
        single_rank = kex.SingleRank()

        for document in self.documents:
            text = ' '.join(document)
            keywords = single_rank.get_keywords(text, n_keywords=3)
            self.raw_data.append(keywords)

    def build_output_list(self):
        pass


if __name__ == '__main__':
    # stopwords_file = r"D:\Program Files\GithubRepositorySet\NLP\cfsfdp_algo\data\stop_words\cn_stopwords.txt"
    # singlerank = SINGLERANK(stopwords_file=stopwords_file, inputs=read_test_tsv())
    # singlerank.keyword_extraction()
    # print(singlerank.raw_data)
    # print(type(singlerank.raw_data[0][0]))
    model = kex.LexRank()
    sample = '''
    We propose a novel unsupervised keyphrase extraction approach that filters candidate keywords using outlier detection.
    It starts by training word embeddings on the target document to capture semantic regularities among the words. It then
    uses the minimum covariance determinant estimator to model the distribution of non-keyphrase word vectors, under the
    assumption that these vectors come from the same distribution, indicative of their irrelevance to the semantics
    expressed by the dimensions of the learned vector representation. Candidate keyphrases only consist of words that are
    detected as outliers of this dominant distribution. Empirical results show that our approach outperforms state
    of-the-art and recent unsupervised keyphrase extraction methods.
    '''
    print(model.get_keywords(sample, n_keywords=2))
