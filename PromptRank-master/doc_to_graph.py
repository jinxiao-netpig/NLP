import string
from collections import Counter

import networkx as nx
import nltk
from nltk.corpus import stopwords


def create_word_network(text):
    porter = nltk.PorterStemmer()
    # 创建空的无向图
    G = nx.Graph()

    # 获取停用词列表
    stop_words = set(stopwords.words('english'))

    # 移除标点符号
    translator = str.maketrans('', '', string.punctuation)
    words = text.translate(translator).split()

    # 过滤停用词和空字符串
    filtered_words = [word for word in words if word.lower() not in stop_words and word != '']

    # 词形还原
    stemmed_words = [porter.stem(word) for word in filtered_words]

    # 计算单词频率
    word_freq = Counter(stemmed_words)

    # 添加节点及其频率属性
    for word in stemmed_words:
        G.add_node(word, frequency=word_freq[word])
    # 遍历过滤后的单词列表并添加边
    for i in range(len(stemmed_words) - 1):
        G.add_edge(stemmed_words[i], stemmed_words[i + 1])

    return G
