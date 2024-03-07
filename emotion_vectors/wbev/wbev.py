import jieba
import numpy as np
import pandas as pd

jieba.enable_paddle()

emotion_list = {
    "NN": 0,
    "NE": 1,
    "ND": 2,
    "NB":
}


class WBEV:
    def __init__(self, emotion_dictionary: pd.DataFrame):
        self.emotion_dictionary = emotion_dictionary
        emotion_list = {}

    def __get_emotion_vector(self, title: str) -> np.ndarray:
        title_list = np.zeros(21, dtype=float)
        seg_list = jieba.cut(title, cut_all=False)

        for word in seg_list:
            if word in self.emotion_dictionary["词语"]:
                title_list[emotion_list[self.emotion_dictionary.loc[self.emotion_dictionary["词语"] == word, "情感分类"]]] += \
                    self.emotion_dictionary.loc[self.emotion_dictionary["词语"] == word, "强度"]

        return title_list

    def get_emotion_vector(self, title: str) -> np.ndarray:
        return self.__get_emotion_vector(title)

    def __build_emotion_list(self):
        emotions = self.emotion_dictionary["情感分类"]
        index = 1

        for word in emotions:
            if word not in
