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

        token_embeddings = []
        for token_i in range(len(tokenized_text)):
            print("token_i: {}".format(token_i))
            print("len(tokenized_text): {}".format(len(tokenized_text)))
            hidden_layers = []
            for layer_i in range(len(encoded_layers)):
                print("layer_i: {}".format(layer_i))
                print("len(encoded_layers[layer_i]): {}".format(len(encoded_layers[layer_i])))
                print("len(encoded_layers[layer_i][0]): {}".format(len(encoded_layers[layer_i][0])))
                vec = encoded_layers[layer_i][0][token_i]
                hidden_layers.append(vec)
            token_embeddings.append(hidden_layers)

        # 句子向量
        sentence_embedding = torch.mean(encoded_layers[11], 1).numpy()
        self.set_embedding(sentence_embedding)


if __name__ == '__main__':
    # text = "I want to eat a apple."
    # bert_model = BertToEmbedding()
    # bert_model.text_to_embedding(text=text)
    # print(bert_model.get_embedding())
    text = " the man went to the store "
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    model = BertModel.from_pretrained("bert-base-cased")
    tokenized_text = tokenizer.tokenize(text)  # token初始化
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)  # 获取词汇表索引
    tokens_tensor = torch.tensor([indexed_tokens])  # 将输入转化为torch的tensor
    with torch.no_grad():  # 禁用梯度计算 因为只是前向传播获取隐藏层状态，所以不需要计算梯度
        last_hidden_states = model(tokens_tensor)[0]
    token_embeddings = []
    for token_i in range(len(tokenized_text)):
        hidden_layers = []
        for layer_i in range(len(last_hidden_states)):
            vec = last_hidden_states[layer_i][0][token_i]  # 如果输入是单句不分块中间是0，因为只有一个维度，如果分块还要再遍历一次
            hidden_layers.append(vec)
        token_embeddings.append(hidden_layers)

    sentence_embedding = torch.mean(last_hidden_states[11], 1).numpy()
    print(sentence_embedding)

