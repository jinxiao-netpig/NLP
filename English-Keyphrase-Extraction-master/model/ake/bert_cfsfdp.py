import logging
import time
from typing import TypeVar

import kex
import nltk
import numpy as np
import spacy
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords

from model.ake.meta_method import MetaMethod
from model.other.bert_to_embedding import BertToEmbedding
from model.other.cfsfdp import CFSFDP
from model.other.nltk_sense_to_embedding import NltkSenseToEmbedding

T = TypeVar('T')


class BERTCFSFDP(MetaMethod):
    def __init__(self, epsilon: float, threshold: float):
        self.epsilon = epsilon  # 截断距离
        self.threshold = threshold  # 密度峰值阈值
        self.points = {}  # 数据点名称:数据点坐标
        self.cfsfdp = CFSFDP(epsilon=epsilon, threshold=threshold, points={})  # cfsfdp聚类器
        self.bert_model = BertToEmbedding()  # bert编码器
        self.sen_to_words = spacy.load('en_core_web_sm')  # 分词器
        self.lemmatizer = WordNetLemmatizer()  # 词干还原器
        self.sence_encoder = NltkSenseToEmbedding()  # 情绪向量编码器
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

        # 文本预处理，并存入 self.points
        self.preprocess_text(text=text)

        # 对词向量进行聚类
        self.cfsfdp.set_points(self.points)
        self.cfsfdp.fit()
        centers = self.cfsfdp.center_indices_list
        centers_list = [""] * len(centers)
        for key, value in centers.items():
            centers_list[value - 1] = key

        if n_keywords > len(centers_list):
            n_keywords = len(centers_list)

        return centers_list[:n_keywords]

    def preprocess_text(self, text: str):
        word_to_embedding: dict[str, np.ndarray] = {}

        # 划分句子
        texts = text.split(sep=".")
        for sentence in texts:
            # 获得句子向量
            sentence_embedding = self.bert_model.text_to_embedding(sentence)
            # 将句子拆成单词，并进行停用词过滤以及词干还原
            words = nltk.word_tokenize(sentence)
            words_set = set(words)
            words_set_lower = [word.lower() for word in words_set if word.isalpha()]
            stopword_set = set(stopwords.words('english'))
            words_set_with_stopword = [word for word in words_set_lower if word not in stopword_set]
            # 词干还原
            word_stems = []
            for word in words_set_with_stopword:
                doc = self.sen_to_words(word)
                lemmatized_word = doc[0].lemma_
                word_stems.append(lemmatized_word)

            # 将句子拆成词组
            phrases = []
            doc = self.sen_to_words(sentence)
            for chunk in doc.noun_chunks:
                phrases.append(chunk.text)

            words_phrases_list = word_stems + phrases
            words_phrases_set = set(words_phrases_list)

            # 获得词向量
            for word in words_phrases_set:
                word_embedding = self.bert_model.text_to_embedding(word)
                sense_embedding = self.sence_encoder.text_to_embedding(word)
                # 词向量+句子向量+情感向量
                word_sen_sense_embedding = np.concatenate((word_embedding, sentence_embedding, sense_embedding))
                word_to_embedding[word] = word_sen_sense_embedding

        self.points = word_to_embedding

    def train_model(self, learning_rate: float):
        epoch = 0
        # 加载数据
        self.cfsfdp.set_points(points=self.points)
        while epoch < 500:
            epoch += 1
            # 训练前打印参数信息
            logging.info("start epoch: {}".format(epoch))
            logging.info("epsilon: {}".format(self.epsilon))
            logging.info("threshold: {}".format(self.threshold))
            self.cfsfdp.set_epsilon(self.epsilon)
            self.cfsfdp.set_threshold(self.threshold)
            # 开始计算聚类中心
            self.cfsfdp.fit()

            centers_num = len(self.cfsfdp.center_indices_list)
            loss = centers_num - 6
            logging.info("centers_num: {}".format(centers_num))
            logging.info("loss: {}".format(loss))
            # 损失提前收敛就直接退出循环
            if 0 <= loss <= 2:
                break

            # 更新 threshold 参数，值越大，聚类数量越少
            self.threshold = self.threshold + loss * learning_rate
            logging.info("-------------------------------------------------------------")


if __name__ == '__main__':
    # bert_cfsfdp_model = BERTCFSFDP(epsilon=1, threshold=3)
    # bert_cfsfdp_model.keyword_extraction("Inspec")
    # bert_cfsfdp_model.show_output_list()

    nlp = spacy.load('en_core_web_sm')
    test_word_list = ["men", "computers", "ate", "running", "fancier"]
    lemmatized_list = []
    for word in test_word_list:
        doc = nlp(word)
        lemmatized_word = doc[0].lemma_
        lemmatized_list.append(lemmatized_word)
    print("lemmatized_list: {}".format(lemmatized_list))
