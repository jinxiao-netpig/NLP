import logging
from abc import abstractmethod

import numpy as np


class ToEmbedding:
    embedding: np.ndarray

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def text_to_embedding(self, text: str):
        """
        将输入文本转为 embedding

        :param text: 输入的文本
        :return:
        """

        logging.info("MODEL: " + str(type(self)) + ", " + "OPERATION: " + "text_to_embedding")
        pass

    def load_model(self):
        pass

    def train_model(self):
        pass

    def get_embedding(self) -> np.ndarray:
        return self.embedding

    def set_embedding(self, embedding: np.ndarray):
        self.embedding = embedding
        pass
