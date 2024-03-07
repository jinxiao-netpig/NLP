import pandas


def load_emotion_dictionary():
    df = pandas.read_csv("../情感词.csv", usecols=[0, 1, 4, 5])

    return df


if __name__ == '__main__':
    print(load_emotion_dictionary())
