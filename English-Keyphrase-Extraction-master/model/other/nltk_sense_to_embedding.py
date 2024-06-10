import numpy
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from model.other.to_embedding import ToEmbedding


class NltkSenseToEmbedding(ToEmbedding):
    def __init__(self):
        pass

    def text_to_embedding(self, text: str) -> np.ndarray:
        # nltk.download('vader_lexicon')
        sid = SentimentIntensityAnalyzer()
        # 进行情感分析
        sentiment_scores = sid.polarity_scores(text)
        embedding_list = []

        for score in sentiment_scores.values():
            embedding_list.append(score)

        embedding = numpy.array(embedding_list)
        # self.set_embedding(embedding)

        return embedding


if __name__ == '__main__':
    # nltk.download('vader_lexicon')
    # 初始化 VADER 情感分析工具
    sid = SentimentIntensityAnalyzer()

    # 示例文本
    text = "I am so bad"

    # 进行情感分析
    sentiment_scores = sid.polarity_scores(text)

    # 打印情感分析结果
    print(sentiment_scores)
