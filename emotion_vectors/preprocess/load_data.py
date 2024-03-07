import pandas


def load_emotion_dictionary():
    df = pandas.read_csv("../情感词.csv", usecols=[0, 1, 2, 3, 4, 5, 6])

    return df


if __name__ == '__main__':
    print(load_emotion_dictionary())
