import time
from typing import TypeVar

import kex
import numpy as np
import spacy

from model.ake.meta_method import MetaMethod
from model.other.bert_to_embedding import BertToEmbedding
from model.other.cfsfdp import CFSFDP

T = TypeVar('T')


class BERTCFSFDP(MetaMethod):
    def __init__(self, epsilon: float, threshold: float, points: dict[T, np.ndarray]):
        self.epsilon = epsilon
        self.threshold = threshold
        self.points = points
        self.cfsfdp = CFSFDP(epsilon=epsilon, threshold=threshold, points=points)  # 聚类
        self.bert_model = BertToEmbedding()  # bert编码器
        self.sen_to_words = spacy.load('en_core_web_sm')  # 分词器
        pass

    def keyword_extraction(self, dataset_name: str):
        super().keyword_extraction(dataset_name)
        time1 = time.time()

        # 测试集取前500条数据
        size = 500
        json_line, _ = kex.get_benchmark_dataset(dataset_name)
        if size > len(json_line):
            size = len(json_line)
        json_line = json_line[:size]

        for line in json_line:
            text = line['source']
            results = self.__keyword_extraction(text, n_keywords=3)
            predict_keywords = []
            # 构建结果
            for result in results:
                predict_keywords.append(result)
            self.output_list[";".join(line['keywords'])] = predict_keywords

        time2 = time.time()
        self.cost = int(time2 - time1)

    def __keyword_extraction(self, text: str, n_keywords: int) -> list[str]:
        """
        具体对每条文本数据进行关键词提取操作

        :param text: 文本数据
        :param n_keywords: 提取关键词的个数
        :return: 关键词列表
        """

        keywords = []

        # 划分句子
        texts = text.split(sep=".")
        for sentence in texts:
            # 获得句子向量
            sentence_embedding = self.bert_model.text_to_embedding(sentence)
            # 将句子拆成词，并进行停用词过滤以及词干还原
        pass

    def train_model(self):
        pass


if __name__ == '__main__':
    title = 'Personalized Leather Case, Monogram Engraved Phone Case for iPhone 14 Pro, Crossbody Phone Case with Card Solt for iPhone13 12 11 Max, MINI'
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(title)
    noun_phrases = []
    for chunk in doc.noun_chunks:
        noun_phrases.append(chunk.text)
    print(noun_phrases)
