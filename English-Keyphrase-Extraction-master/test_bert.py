import torch
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer, BertModel

if __name__ == '__main__':

    # # 下载nltk所需的资源
    # nltk.download('punkt')

    # 定义函数进行分词和编码
    def tokenize_and_encode(sentence):
        # 使用nltk进行分词
        words = word_tokenize(sentence)

        # 加载BERT模型和分词器
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')

        # 对每个词进行编码
        input_ids = tokenizer(words, return_tensors='pt', is_split_into_words=True)

        # 获取BERT模型的输出
        with torch.no_grad():
            outputs = model(**input_ids)

        # 提取最后一层的隐藏状态
        last_hidden_states = outputs.last_hidden_state

        # 打印每个词的编码
        for word, vector in zip(words, last_hidden_states[0]):
            print(f"Word: {word}")
            print(f"Vector: {vector[:5]}...")  # 打印前5个值以示例
            print()


    # 输入句子
    sentence = "The quick brown fox jumps over the lazy dog."

    # 分词和编码
    tokenize_and_encode(sentence)
