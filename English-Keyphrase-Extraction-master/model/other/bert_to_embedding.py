import numpy as np
import torch
from transformers import BertTokenizer, BertModel

from model.other.to_embedding import ToEmbedding


# Tokenizer 负责把输入的文本做切分然后变成向量
# Model 负责根据输入向量提取语义信息


class BertToEmbedding(ToEmbedding):
    def __init__(self):
        self.model_name = "bert-base-cased"

    def text_to_embedding(self, text: str) -> np.ndarray:
        super().text_to_embedding(text=text)

        # 加载BERT模型和BERT分词器
        tokenizer = BertTokenizer.from_pretrained(self.model_name)
        model = BertModel.from_pretrained(self.model_name)

        # 对文本进行分词和转换
        input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])

        # 通过BERT模型获取最后一层隐藏状态作为embedding向量
        with torch.no_grad():
            outputs = model(input_ids)
            embedding_vector = outputs[0][0]

        # self.set_embedding(embedding_vector[0].numpy())

        return embedding_vector[0].numpy()


if __name__ == '__main__':
    # # 加载BERT模型和分词器
    # tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    # model = BertModel.from_pretrained('bert-base-cased')
    #
    # # 定义要转换的英文文本
    # text = "This is a sample text to be converted into an embedding vector"
    #
    # # 对文本进行分词和转换
    # input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])
    # print("input_ids length: {}".format(len(input_ids)))
    #
    # # 通过BERT模型获取最后一层隐藏状态作为embedding向量
    # with torch.no_grad():
    #     outputs = model(input_ids)
    #     embedding_vector = outputs[0][0]
    #
    # print("embedding_vector: {}".format(embedding_vector))
    # print("embedding_vector type: {}".format(type(embedding_vector)))
    # print("embedding_vector length: {}".format(len(embedding_vector)))
    # print("embedding_vector[0] length: {}".format(len(embedding_vector[0])))
    # print("embedding_vector[0] type: {}".format(type(embedding_vector[0])))

    model = BertToEmbedding()
    text1 = "a good man."
    text2 = "a bad woman."


    def cosine_similarity(A, B):
        # 计算点积
        dot_product = np.dot(A, B)
        # 计算范数（即向量长度）
        norm_A = np.linalg.norm(A)
        norm_B = np.linalg.norm(B)
        # 计算余弦相似度
        cosine_sim = dot_product / (norm_A * norm_B)
        return cosine_sim


    def euclidean_distance(A, B):
        # 计算向量之差
        diff = A - B
        # 计算差的平方
        sq_diff = np.square(diff)
        # 计算平方和
        sum_sq_diff = np.sum(sq_diff)
        # 计算平方根
        distance = np.sqrt(sum_sq_diff)
        return distance


    em1 = model.text_to_embedding(text1)
    em2 = model.text_to_embedding(text2)

    sim = euclidean_distance(em1, em2)
    print("sim: {}".format(sim))  # sim: 4.071786880493164
    # print("em1: {}".format(em1))
    # print("em2: {}".format(em2))
    print("type(sim): {}".format(type(sim)))
