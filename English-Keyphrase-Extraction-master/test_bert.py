if __name__ == '__main__':
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    import string
    import torch
    from transformers import BertTokenizer, BertModel

    # 下载nltk所需的资源
    nltk.download('punkt')
    nltk.download('stopwords')

    # 加载预训练的BERT模型和分词器
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # 输入句子
    sentence = "The quick brown fox jumps over the lazy dog."

    # 对句子进行分词
    words = word_tokenize(sentence)

    # 加载停用词和标点符号
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)

    # 过滤停用词和标点符号
    filtered_words = [word for word in words if word.lower() not in stop_words and word not in punctuation]

    # 对过滤后的词进行编码
    encoded_input = tokenizer(filtered_words, return_tensors='pt', is_split_into_words=True, add_special_tokens=False)

    # 获取BERT模型的输出
    with torch.no_grad():
        outputs = model(**encoded_input)

    # 提取最后一层的隐藏状态
    last_hidden_states = outputs.last_hidden_state

    # 打印每个词的词向量
    for word, vector in zip(filtered_words, last_hidden_states[0]):
        print(f"Word: {word}")
        print(f"Vector: {vector[:10]}...")  # 打印前5个值以示例
        print()
