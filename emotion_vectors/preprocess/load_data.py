import pandas as pd


def load_emotion_dictionary() -> pd.DataFrame:
    """
    加载情感字典

    词语  词性种类    情感分类    强度

    :return: 情感词汇集合
    """

    df = pd.read_csv("../情感词.csv", usecols=[0, 1, 4, 5])

    return df


if __name__ == '__main__':
    print(load_emotion_dictionary())
