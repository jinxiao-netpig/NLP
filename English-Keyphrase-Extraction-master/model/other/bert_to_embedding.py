import nltk
from transformers import BertTokenizer, BertModel

from model.other.to_embedding import ToEmbedding


# Tokenizer 负责把输入的文本做切分然后变成向量
# Model 负责根据输入向量提取语义信息


class BertToEmbedding(ToEmbedding):
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = "bert-base-cased"

    def load_model(self):
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertModel.from_pretrained(self.model_name)

    def text_to_embedding(self, text: str):
        super().text_to_embedding(text=text)


if __name__ == '__main__':
    text = "I want to eat a apple."
    tokens = nltk.word_tokenize(text)
    print(tokens)
