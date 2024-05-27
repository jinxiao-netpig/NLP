import nltk
import torch
from transformers import BertTokenizer, BertModel

from model.other.to_embedding import ToEmbedding


# Tokenizer 负责把输入的文本做切分然后变成向量
# Model 负责根据输入向量提取语义信息


class BertToEmbedding(ToEmbedding):
    def __init__(self):
        self.model_name = "bert-base-cased"

    def text_to_embedding(self, text: str):
        super().text_to_embedding(text=text)

        # 加载BERT模型和BERT分词器
        bert_tokenizer = BertTokenizer.from_pretrained(self.model_name)
        bert_model = BertModel.from_pretrained(self.model_name)

        marked_text = "[CLS] " + text + " [SEP]"
        tokenized_text = bert_tokenizer.tokenize(marked_text)
        indexed_tokens = bert_tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        # 将模型置于评估模式，而不是训练模式。在这种情况下，评估模式关闭了训练中使用的dropout正则化。
        bert_model.eval()
        # 禁用梯度计算，节省内存，并加快计算速度
        with torch.no_grad():
            encoded_layers, _ = bert_model(tokens_tensor, segments_tensors)


if __name__ == '__main__':
    text = "I want to eat a apple."
    tokens = nltk.word_tokenize(text)
    print(tokens)
