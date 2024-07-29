# coding=utf-8
import codecs
import json
import os
import pickle
import re
from typing import Tuple, List

import nltk
from nltk.corpus import stopwords
from stanfordcorenlp import StanfordCoreNLP
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import T5Tokenizer

from vividi.test_main import candidates_to_graph

global_reviews = None

dataName = ""
dataDir = ""

MAX_LEN = None
enable_filter = None
temp_en = None
temp_de = None

StanfordCoreNLP_path = './stanford-corenlp-full-2018-02-27'

stopword_dict = set(stopwords.words('english'))
en_model = StanfordCoreNLP(StanfordCoreNLP_path, quiet=True)
tokenizer = None

GRAMMAR = """  NP:
        {<NN.*|JJ>*<NN.*>}  # Adjective(s)(optional) + Noun(s)"""


def extract_candidates(tokens_tagged, no_subset=False):
    """
    Based on part of speech return a list of candidate phrases
    :param text_obj: Input text Representation see @InputTextObj
    :param no_subset: if true won't put a candidate which is the subset of an other candidate
    :return keyphrase_candidate: list of list of candidate phrases: [tuple(string,tuple(start_index,end_index))]
    """

    cans_count = dict()

    np_parser = nltk.RegexpParser(GRAMMAR)  # Noun phrase parser，一个基于词性的正则解析器RegexpParser，可以通过正则表达式匹配特定标记的词块
    keyphrase_candidate = []
    np_pos_tag_tokens = np_parser.parse(tokens_tagged)
    count = 0
    for token in np_pos_tag_tokens:
        if (isinstance(token, nltk.tree.Tree) and token._label == "NP"):
            np = ' '.join(word for word, tag in token.leaves())
            length = len(token.leaves())
            start_end = (count, count + length)
            count += length

            if len(np.split()) == 1:
                if np not in cans_count.keys():
                    cans_count[np] = 0
                cans_count[np] += 1

            keyphrase_candidate.append((np, start_end))

        else:
            count += 1

    if enable_filter == True:
        i = 0
        while i < len(keyphrase_candidate):
            can, pos = keyphrase_candidate[i]
            # pos[0] > 50 and
            if can in cans_count.keys() and cans_count[can] == 1:
                keyphrase_candidate.pop(i)
                continue
            i += 1

    return keyphrase_candidate


class InputTextObj:
    """Represent the input text in which we want to extract keyphrases"""

    def __init__(self, en_model, text=""):
        """
        :param is_sectioned: If we want to section the text.
        :param en_model: the pipeline of tokenization and POS-tagger
        :param considered_tags: The POSs we want to keep
        """
        self.considered_tags = {'NN', 'NNS', 'NNP', 'NNPS', 'JJ'}

        self.tokens = []
        self.tokens_tagged = []
        self.tokens = en_model.word_tokenize(text)  # 分词
        self.tokens_tagged = en_model.pos_tag(text)  # 获得词性
        assert len(self.tokens) == len(self.tokens_tagged)
        for i, token in enumerate(self.tokens):  # 过滤停用词
            if token.lower() in stopword_dict:
                self.tokens_tagged[i] = (token, "IN")
        self.keyphrase_candidate = extract_candidates(self.tokens_tagged, en_model)


class KPE_Dataset(Dataset):
    def __init__(self, docs_pairs):
        self.docs_pairs = docs_pairs
        self.total_examples = len(self.docs_pairs)

    def __len__(self):
        return self.total_examples

    def __getitem__(self, idx):
        doc_pair = self.docs_pairs[idx]
        en_input_ids = doc_pair[0][0]
        en_input_mask = doc_pair[1][0]
        de_input_ids = doc_pair[2][0]
        dic = doc_pair[3]

        return [en_input_ids, en_input_mask, de_input_ids, dic]


def clean_text(text="", database="Inspec"):
    # Specially for Duc2001 Database
    if (database == "Duc2001" or database == "Semeval2017"):
        pattern2 = re.compile(r'[\s,]' + '[\n]{1}')
        while (True):
            if (pattern2.search(text) is not None):
                position = pattern2.search(text)
                start = position.start()
                end = position.end()
                # start = int(position[0])
                text_new = text[:start] + "\n" + text[start + 2:]
                text = text_new
            else:
                break

    pattern2 = re.compile(r'[a-zA-Z0-9,\s]' + '[\n]{1}')
    while (True):
        if (pattern2.search(text) is not None):
            position = pattern2.search(text)
            start = position.start()
            end = position.end()
            # start = int(position[0])
            text_new = text[:start + 1] + " " + text[start + 2:]
            text = text_new
        else:
            break

    pattern3 = re.compile(r'\s{2,}')
    while (True):
        if (pattern3.search(text) is not None):
            position = pattern3.search(text)
            start = position.start()
            end = position.end()
            # start = int(position[0])
            text_new = text[:start + 1] + "" + text[start + 2:]
            text = text_new
        else:
            break

    pattern1 = re.compile(r'[<>[\]{}]')
    text = pattern1.sub(' ', text)
    text = text.replace("\t", " ")
    text = text.replace(' p ', '\n')
    text = text.replace(' /p \n', '\n')
    lines = text.splitlines()
    # delete blank line
    text_new = ""
    for line in lines:
        if (line != '\n'):
            text_new += line + '\n'

    return text_new


def get_long_data(file_path="data/nus/nus_test.json"):
    """ Load file.jsonl ."""
    data = {}
    labels = {}
    with codecs.open(file_path, 'r', 'utf-8') as f:
        json_text = f.readlines()
        for i, line in tqdm(enumerate(json_text), desc="Loading Doc ..."):
            try:
                jsonl = json.loads(line)
                keywords = jsonl['keywords'].lower().split(";")
                abstract = jsonl['abstract']
                fulltxt = jsonl['fulltext']
                doc = ' '.join([abstract, fulltxt])
                doc = re.sub('\. ', ' . ', doc)
                doc = re.sub(', ', ' , ', doc)

                doc = clean_text(doc, database="nus")
                doc = doc.replace('\n', ' ')
                data[jsonl['name']] = doc
                labels[jsonl['name']] = keywords
            except:
                raise ValueError
    return data, labels


def get_TripAdvisor_data(file_path="data/TripAdvisor/reviews.pickle"):
    data = {}
    labels = {}
    assert os.path.exists(file_path)
    reviews = pickle.load(open(file_path, 'rb'))
    global global_reviews
    # reviews 是一个 list
    global_reviews = reviews

    for i, review in tqdm(enumerate(reviews), desc="Loading Doc ..."):
        # fea 只有一个，是个 str
        (fea, adj, tem, sco) = review['template']  # feature, opinion, template, sentiment
        doc = tem
        doc = re.sub('\. ', ' . ', doc)
        doc = re.sub(', ', ' , ', doc)
        doc = doc.replace('\n', ' ')
        data[i] = doc
        labels[i] = [fea]
    return data, labels


def get_short_data(file_path="data/kp20k/kp20k_valid2k_test.json"):
    """ Load file.jsonl ."""
    data = {}
    labels = {}
    with codecs.open(file_path, 'r', 'utf-8') as f:
        json_text = f.readlines()
        for i, line in tqdm(enumerate(json_text), desc="Loading Doc ..."):
            try:
                jsonl = json.loads(line)
                keywords = jsonl['keywords'].lower().split(";")
                abstract = jsonl['abstract']
                doc = abstract
                doc = re.sub('\. ', ' . ', doc)
                doc = re.sub(', ', ' , ', doc)

                doc = clean_text(doc, database="kp20k")
                doc = doc.replace('\n', ' ')
                data[i] = doc
                labels[i] = keywords
            except:
                raise ValueError
    return data, labels


def get_duc2001_data(file_path="data/DUC2001"):
    pattern = re.compile(r'<TEXT>(.*?)</TEXT>', re.S)
    data = {}
    labels = {}
    for dirname, dirnames, filenames in os.walk(file_path):
        for fname in filenames:
            if (fname == "annotations.txt"):
                # left, right = fname.split('.')
                infile = os.path.join(dirname, fname)
                f = open(infile, 'rb')
                text = f.read().decode('utf8')
                lines = text.splitlines()
                for line in lines:
                    left, right = line.split("@")
                    d = right.split(";")[:-1]
                    l = left
                    labels[l] = d
                f.close()
            else:
                infile = os.path.join(dirname, fname)
                f = open(infile, 'rb')
                text = f.read().decode('utf8')
                text = re.findall(pattern, text)[0]

                text = text.lower()
                text = clean_text(text, database="Duc2001")
                data[fname] = text.strip("\n")
                # data[fname] = text
    return data, labels


def get_inspec_data(file_path="data/Inspec"):
    data = {}  # 文件id:原始文本内容
    labels = {}  # 文件id:关键词
    for dirname, dirnames, filenames in os.walk(file_path):
        for fname in filenames:
            left, right = fname.split('.')  # 文件名组成：文件id.文件内容类别
            if (right == "abstr"):
                infile = os.path.join(dirname, fname)
                f = open(infile)
                text = f.read()
                text = text.replace("%", '')
                text = clean_text(text)
                data[left] = text
            if (right == "uncontr"):
                infile = os.path.join(dirname, fname)
                f = open(infile)
                text = f.read()
                text = text.replace("\n", ' ')
                text = clean_text(text, database="Inspec")
                text = text.lower()
                label = text.split("; ")
                labels[left] = label
    return data, labels


def get_semeval2017_data(data_path="data/SemEval2017/docsutf8", labels_path="data/SemEval2017/keys"):
    data = {}
    labels = {}
    for dirname, dirnames, filenames in os.walk(data_path):
        for fname in filenames:
            left, right = fname.split('.')
            infile = os.path.join(dirname, fname)
            # f = open(infile, 'rb')
            # text = f.read().decode('utf8')
            with codecs.open(infile, "r", "utf-8") as fi:
                text = fi.read()
                text = text.replace("%", '')
            text = clean_text(text, database="Semeval2017")
            data[left] = text.lower()
            # f.close()
    for dirname, dirnames, filenames in os.walk(labels_path):
        for fname in filenames:
            left, right = fname.split('.')
            infile = os.path.join(dirname, fname)
            f = open(infile, 'rb')
            text = f.read().decode('utf8')
            text = text.strip()
            ls = text.splitlines()
            labels[left] = ls
            f.close()
    return data, labels


def remove(text):
    text_len = len(text.split())
    remove_chars = '[’!"#$%&\'()*+,./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    text = re.sub(remove_chars, '', text)
    re_text_len = len(text.split())
    if text_len != re_text_len:
        return True
    else:
        return False


def generate_doc_pairs(doc, candidates: List[Tuple[str, Tuple[int, int]]], idx):
    count = 0
    doc_pairs = []

    en_input = tokenizer(doc, max_length=MAX_LEN, padding="max_length", truncation=True, return_tensors="pt")
    en_input_ids = en_input["input_ids"]
    en_input_mask = en_input["attention_mask"]

    # todo: 在这里把图节点的信息放进来就行
    G = candidates_to_graph(candidates=candidates)

    for id, can_and_pos in enumerate(candidates):
        # 这里其实就已经把所有的输出都准备好了，并且已经编码了
        candidate = can_and_pos[0]
        # Remove stopwords in a candidate
        if remove(candidate):
            count += 1
            continue

        de_input = temp_de + candidate + " ."
        de_input_ids = tokenizer(de_input, max_length=30, padding="max_length", truncation=True, return_tensors="pt")[
            "input_ids"]
        de_input_ids[0, 0] = 0
        de_input_len = (de_input_ids[0] == tokenizer.eos_token_id).nonzero()[0].item() - 2

        #         for i in de_input_ids[0]:
        #             print(tokenizer.decode(i))
        #         print(de_input_len)

        #         x = tokenizer(temp_de, return_tensors="pt")["input_ids"]
        #         for i in x[0]:
        #             print(tokenizer.decode(i))
        #         exit(0)
        dic = {"de_input_len": de_input_len,
               "candidate": candidate,
               "idx": idx,
               "pos": can_and_pos[1][0],
               "candidate_node_score": G.nodes[candidate]['weight']}

        doc_pairs.append([en_input_ids, en_input_mask, de_input_ids, dic])
        # print(tokenizer.decode(en_input_ids[0]))
        # print(tokenizer.decode(de_input_ids[0]))
        # print(candidate)
        # print(de_input_len)
        # print()
        # exit(0)
    return doc_pairs, count


def init(setting_dict):
    '''
    Init template, max length and tokenizer.
    '''

    global MAX_LEN, temp_en, temp_de, tokenizer, enable_filter
    MAX_LEN = setting_dict["max_len"]
    temp_en = setting_dict["temp_en"]
    temp_de = setting_dict["temp_de"]
    enable_filter = setting_dict["enable_filter"]

    tokenizer = T5Tokenizer.from_pretrained(
        r"C:\Users\HP\.cache\huggingface\hub\models--t5-base\snapshots\a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1",
        model_max_length=MAX_LEN)


def data_process(setting_dict, dataset_dir, dataset_name):
    """
    Core API in data.py which returns the dataset

    :return: 包装过的数据集对象，文档列表，文档对应的关键词集合的列表，词形还原处理过的关键词集合的列表
    """

    init(setting_dict)

    # data 是一个字典，值是一个数据集中一条训练数据的已经处理好的原始文本
    if dataset_name == "SemEval2017":
        data, referneces = get_semeval2017_data(dataset_dir + "/docsutf8", dataset_dir + "/keys")
    elif dataset_name == "DUC2001":
        data, referneces = get_duc2001_data(dataset_dir)
    elif dataset_name == "nus":
        data, referneces = get_long_data(dataset_dir + "/nus_test.json")
    elif dataset_name == "krapivin":
        data, referneces = get_long_data(dataset_dir + "/krapivin_test.json")
    elif dataset_name == "kp20k":
        data, referneces = get_short_data(dataset_dir + "/kp20k_valid200_test.json")
    elif dataset_name == "SemEval2010":
        data, referneces = get_short_data(dataset_dir + "/semeval_test.json")
    # todo 添加读peter的那三个数据集
    elif dataset_name == "TripAdvisor":
        data, referneces = get_TripAdvisor_data(dataset_dir + "/reviews.pickle")
    elif dataset_name == "Yelp":
        data, referneces = get_TripAdvisor_data(dataset_dir + "/reviews.pickle")
    elif dataset_name == "Amazon":
        data, referneces = get_TripAdvisor_data(dataset_dir + "/MoviesAndTV/reviews.pickle")
    else:
        data, referneces = get_inspec_data(dataset_dir)

    docs_pairs = []
    doc_list = []  # 原始文本列表
    labels = []
    labels_stemed = []  # 词形还原后的关键词
    t_n = 0
    candidate_num = 0
    porter = nltk.PorterStemmer()

    # data元素的索引，文件id，摘要内容
    for idx, (key, doc) in enumerate(data.items()):

        # Get stemmed labels and document segments
        labels.append([ref.replace(" \n", "") for ref in referneces[key]])  # 处理关键词并放到一个列表里
        labels_s = []
        for l in referneces[key]:
            tokens = l.split()  # 按空格分割
            labels_s.append(' '.join(porter.stem(t) for t in tokens))  # 对关键词进行词形还原

        doc = ' '.join(doc.split()[:MAX_LEN])
        labels_stemed.append(labels_s)
        doc_list.append(doc)

        # Statistic on empty docs
        empty_doc = 0
        try:
            text_obj = InputTextObj(en_model, doc)  # 这个文本对象里有这条文本的候选词，候选词的选取是按照词性和正则来拿到的
        except:
            empty_doc += 1
            print("doc: ", doc)

        # Generate candidates (lower)
        cans = text_obj.keyphrase_candidate
        # todo: 这里就把图构造出来
        candidates = []  # 候选词:词性
        for can, pos in cans:
            if enable_filter == True and len(can.split()) > 4:
                continue
            candidates.append((can.lower(), pos))
        candidate_num += len(candidates)

        # Generate docs_paris for constructing dataset 
        # doc = doc.lower()
        doc = temp_en + "\"" + doc + "\""
        print("doc: ", doc)
        doc_pairs, count = generate_doc_pairs(doc, candidates, idx)  # 这是一条记录文本的所有候选词的输出
        docs_pairs.extend(doc_pairs)  # 这是整个数据集的所有候选词的输出
        t_n += count

    print("candidate_num: ", candidate_num)
    print("unmatched: ", t_n)
    dataset = KPE_Dataset(docs_pairs)  # 把数据集包装成一个对象
    print("examples: ", dataset.total_examples)

    en_model.close()
    return dataset, doc_list, labels, labels_stemed
