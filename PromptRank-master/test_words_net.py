from data import clean_text

if __name__ == '__main__':
    import networkx as nx
    import matplotlib.pyplot as plt
    from nltk.corpus import stopwords
    import string
    import nltk
    from collections import Counter


    # 确保已经下载必要的NLTK资源
    # nltk.download('stopwords')

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


    # 示例文本
    text = "Efficient discovery of grid services is essential for the success of grid computing. The standardization " \
           "of grids based on web services has resulted in the need for scalable web service discovery mechanisms to " \
           "be deployed in grids Even though UDDI has been the de facto industry standard for web-services discovery, " \
           "imposed requirements of tight-replication among registries and lack of autonomous control has severely " \
           "hindered its widespread deployment and usage. With the advent of grid computing the scalability issue of " \
           "UDDI will become a roadblock that will prevent its deployment in grids. In this paper we present our " \
           "distributed web-service discovery architecture, called DUDE (Distributed UDDI Deployment Engine). DUDE " \
           "leverages DHT (Distributed Hash Tables) as a rendezvous mechanism between multiple UDDI registries. DUDE " \
           "enables consumers to query multiple registries, still at the same time allowing organizations to have " \
           "autonomous control over their registries.. Based on preliminary prototype on PlanetLab, we believe that " \
           "DUDE architecture can support effective distribution of UDDI registries thereby making UDDI more robust " \
           "and also addressing its scaling issues. Furthermore, The DUDE architecture for scalable distribution can " \
           "be applied beyond UDDI to any Grid Service Discovery mechanism. "
    text = clean_text(text, database="kp20k")

    # 创建网络
    G = create_word_network(text)

    # 绘制网络
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight="bold")
    plt.show()
