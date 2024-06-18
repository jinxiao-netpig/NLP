import logging
import time
from typing import TypeVar

import kex
import numpy as np
import spacy
import yaml
from nltk import WordNetLemmatizer

from model.ake.meta_method import MetaMethod
from model.other.bert_to_embedding import BertToEmbedding
from model.other.cfsfdp import CFSFDP
from model.other.nltk_sense_to_embedding import NltkSenseToEmbedding

T = TypeVar('T')


# 内存消耗，4G


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
        size = 100
        json_line, _ = kex.get_benchmark_dataset(dataset_name)
        if size > len(json_line):
            size = len(json_line)
        json_line = json_line[:size]

        for line in json_line:
            text = line['source']
            results = self.__keyword_extraction(text, n_keywords=5)
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
            sentence += "."
            dic = self.bert_model.text_to_token_embedding(sentence)
            for word, vector in dic.items():
                word_to_embedding[word] = np.concatenate((vector, self.sence_encoder.text_to_embedding(word)))
            # # # 获得句子向量
            # # sentence_embedding = self.bert_model.text_to_embedding(sentence)
            # # 将句子拆成单词，并进行停用词过滤以及词干还原
            # words = nltk.word_tokenize(sentence)
            # words_set = set(words)
            # words_set_lower = [word.lower() for word in words_set if word.isalpha()]
            # stopword_set = set(stopwords.words('english'))
            # words_set_with_stopword = [word for word in words_set_lower if word not in stopword_set]
            # # 词干还原
            # word_stems = []
            # for word in words_set_with_stopword:
            #     doc = self.sen_to_words(word)
            #     lemmatized_word = doc[0].lemma_
            #     word_stems.append(lemmatized_word)
            #
            # # 将句子拆成词组
            # phrases = []
            # doc = self.sen_to_words(sentence)
            # for chunk in doc.noun_chunks:
            #     phrases.append(chunk.text)
            #
            # words_phrases_list = word_stems + phrases
            # words_phrases_set = set(words_phrases_list)
            #
            # # 获得词向量
            # for word in words_phrases_set:
            #     word_embedding = self.bert_model.text_to_embedding(word)
            #     sense_embedding = self.sence_encoder.text_to_embedding(word)
            #     # # 词向量+句子向量+情感向量
            #     # word_sen_sense_embedding = np.concatenate((word_embedding, sentence_embedding, sense_embedding))
            #     # 词向量+情感向量
            #     word_sen_sense_embedding = np.concatenate((word_embedding, sense_embedding))
            #     word_to_embedding[word] = word_sen_sense_embedding

        self.points = word_to_embedding

    def filter_documents(self, dataset_name: str):
        super().filter_documents(dataset_name)

        json_line, _ = kex.get_benchmark_dataset(dataset_name)
        size = len(json_line)
        test_size = int(size / 5)
        json_line = json_line[test_size:]
        for line in json_line:
            text = line['source']
            self.train_documents.append(text)
        logging.info("train_documents length: " + str(len(self.train_documents)))

    def load_config(self):
        pass

    def dump_config(self):
        logging.info("dump_config -> model/ake/tmp/bert_cfsfdp_config/config.yaml")
        try:
            with open('../configs/config.yaml', 'w', encoding='utf8') as file:
                write_data = {}
                write_data['cfsfdp']['epsilon'] = self.epsilon
                write_data['cfsfdp']['threshold'] = self.threshold
                yaml.dump(write_data, file)
        except Exception as e:
            logging.error("write error: ", e)

    def compute_metric(self):
        """
        计算 P、R、F1

        :return:
        """

        tp_fn_sum = 0
        for oris, preds in self.output_list.items():
            oris_list = oris.split(sep=";")
            for pred in preds:
                if pred in oris:
                    self.tp += 1
                else:
                    self.fp += 1
            tp_fn_sum += len(oris_list)
        self.fn = tp_fn_sum - self.tp

        self.precision = self.tp / (self.tp + self.fp)
        self.recall = self.tp / (self.tp + self.fn)
        self.f_score = 2 * self.precision * self.recall / (self.recall + self.precision)

    def train_model(self, learning_rate: float):
        super().train_model(learn_rate=learning_rate)
        epoch = 0

        while epoch < 100:
            epoch += 1
            # 训练前打印参数信息
            logging.info("start epoch: {}".format(epoch))

            # 一次epoch的训练
            for text in self.train_documents:
                self.preprocess_text(text)
                self.cfsfdp.set_points(points=self.points)
                logging.info("epsilon: {}".format(self.epsilon))
                logging.info("threshold: {}".format(self.threshold))
                self.cfsfdp.set_epsilon(self.epsilon)
                self.cfsfdp.set_threshold(self.threshold)
                # 开始计算聚类中心
                self.cfsfdp.fit()

                centers_num = len(self.cfsfdp.center_indices_list)
                loss = centers_num - 15
                logging.info("words number: {}".format(len(self.points)))
                logging.info("centers_num: {}".format(centers_num))
                logging.info("loss: {}".format(loss))
                # # 损失提前收敛就直接退出循环
                # if 0 <= loss <= 4:
                #     break

                # 更新 epsilon 参数，值越大，聚类数量越少
                self.threshold = self.threshold + loss * learning_rate
            logging.info("-------------------------------------------------------------")

        # dump配置
        self.dump_config()


if __name__ == '__main__':
    # 测试数据
    bert_cfsfdp_model = BERTCFSFDP(epsilon=13, threshold=73)
    bert_cfsfdp_model.keyword_extraction("Inspec")
    bert_cfsfdp_model.compute_metric()
    bert_cfsfdp_model.show_output_list()
    print("bert_cfsfdp_model.precision: {}".format(bert_cfsfdp_model.precision))
    print("bert_cfsfdp_model.recall: {}".format(bert_cfsfdp_model.recall))
    print("bert_cfsfdp_model.f_score: {}".format(bert_cfsfdp_model.f_score))

    # # 训练数据
    # bert_cfsfdp_model = BERTCFSFDP(epsilon=13, threshold=80)
    # bert_cfsfdp_model.filter_documents("Inspec")
    # bert_cfsfdp_model.train_model(learning_rate=0.03)
    # print("epsilon: {}".format(bert_cfsfdp_model.epsilon))
    # print("threshold: {}".format(bert_cfsfdp_model.threshold))
