import torch
from transformers import BertTokenizer, BertModel

if __name__ == '__main__':
    # 加载预训练的BERT模型和tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = BertModel.from_pretrained('bert-base-cased')


    def get_sentence_embedding(sentence):
        # 将句子编码为BERT输入格式
        inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=512)

        # 将输入传递给BERT模型
        with torch.no_grad():
            outputs = model(**inputs)

        # 提取最后一个隐藏层的输出
        last_hidden_states = outputs.last_hidden_state

        # 计算句子的embedding（例如，通过对所有token的embedding取平均值）
        sentence_embedding = torch.mean(last_hidden_states, dim=1)

        return sentence_embedding[0].numpy()


    # 示例句子
    sentence = "The quick brown fox jumps over the lazy dog."

    # 获取句子的embedding
    embedding = get_sentence_embedding(sentence)

    print("Sentence Embedding:", embedding)
    print("Embedding Shape:", embedding.shape)
    print("Embedding Type:", type(embedding))
