if __name__ == '__main__':
    from nltk.tokenize import word_tokenize
    from nltk import pos_tag, ne_chunk
    from nltk.chunk import tree2conlltags
    from transformers import T5Tokenizer, T5Model
    import torch


    # # 下载nltk所需的资源
    # nltk.download('punkt')
    # nltk.download('maxent_ne_chunker')
    # nltk.download('words')
    # nltk.download('averaged_perceptron_tagger')

    def extract_phrases(sentence):
        # 分词
        words = word_tokenize(sentence)

        # POS标注
        tagged = pos_tag(words)

        # 使用nltk的命名实体识别器进行分块
        chunked = ne_chunk(tagged)

        # 提取名词短语（NP）
        iob_tagged = tree2conlltags(chunked)
        phrases = []
        current_phrase = []

        for word, pos, chunk in iob_tagged:
            if chunk.startswith('B'):
                if current_phrase:
                    phrases.append(" ".join(current_phrase))
                    current_phrase = []
                current_phrase.append(word)
            elif chunk.startswith('I'):
                current_phrase.append(word)
            else:
                if current_phrase:
                    phrases.append(" ".join(current_phrase))
                    current_phrase = []
                phrases.append(word)

        if current_phrase:
            phrases.append(" ".join(current_phrase))

        return phrases


    def tokenize_and_encode_phrases(sentence):
        # 提取词组
        phrases = extract_phrases(sentence)

        # 加载BERT模型和分词器
        tokenizer = T5Tokenizer.from_pretrained('t5-base')
        model = T5Model.from_pretrained('t5-base')

        for phrase in phrases:
            # 对每个词组进行编码
            input_ids = tokenizer(phrase, return_tensors='pt')

            # 获取BERT模型的输出
            with torch.no_grad():
                outputs = model(**input_ids)

            # 提取最后一层的隐藏状态
            last_hidden_states = outputs.last_hidden_state

            # 打印每个词组的编码
            print(f"Phrase: {phrase}")
            print(f"Vector: {last_hidden_states[0][0][:5]}...")  # 打印前5个值以示例
            print()


    # 输入句子
    sentence = "There is a apple tree, and I like eating the red apple."

    # 分词组和编码
    tokenize_and_encode_phrases(sentence)
