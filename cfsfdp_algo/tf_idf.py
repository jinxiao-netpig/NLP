from typing import Union

import jieba
import jieba.analyse
import numpy as np
from pandas.core.arrays import ExtensionArray


class TFIDF:
    output_list = {}  # 原始关键词:预测关键词

    def __init__(self, inputs: tuple[Union[ExtensionArray, np.ndarray], Union[ExtensionArray, np.ndarray]]):
        self.inputs = inputs

    def keyword_extraction(self):
        # jieba.analyse.set_stop_words('停用词库.txt')
        for text in enumerate(self.inputs[0].tolist()):
            keywords = jieba.analyse.extract_tags(text[1], topK=3)
            self.output_list[self.inputs[1].tolist()[text[0]]] = keywords

# if __name__ == '__main__':
#     td_idf = TFIDF(read_test_tsv())
#     td_idf.keyword_extraction()
#     print(td_idf.output_list)
