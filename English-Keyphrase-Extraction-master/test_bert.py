import time

import torch
from transformers import BertTokenizer, BertModel

if __name__ == '__main__':
    time1 = time.time()

    # 加载预训练的BERT模型和分词器
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = BertModel.from_pretrained('bert-base-cased')

    # 输入句子
    sentence = "The quick brown fox jumps over the lazy dog."

    # 对句子进行分词，并添加特殊标记 [CLS] 和 [SEP]
    input_ids = tokenizer.encode(sentence, return_tensors='pt')

    # 获取BERT模型的输出
    with torch.no_grad():
        outputs = model(input_ids)

    # 提取最后一层的隐藏状态
    last_hidden_states = outputs.last_hidden_state

    # 解码分词器的标记以匹配编码表示
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    time2 = time.time()
    print("time cost: {}".format(time2 - time1))

    # 打印每个词的编码
    for token, vector in zip(tokens, last_hidden_states[0]):
        print(f"Token: {token}")
        print("vector length: {}".format(len(vector)))
        print(f"Vector: {vector[:5]}...")  # 打印前5个值以示例
        print()
