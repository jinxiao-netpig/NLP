import logging
import time

import kex

from model.ake.meta_method import MetaMethod


class SingleTPR(MetaMethod):
    def __init__(self):
        pass

    def filter_documents(self, dataset_name: str):
        super().filter_documents(dataset_name)

        json_line, _ = kex.get_benchmark_dataset(dataset_name)
        size = len(json_line)
        test_size = int(size / 5)
        json_line = json_line[test_size:]
        for line in json_line:
            text = line['source']
            self.train_documents.append(text)
        logging.info("train_documents length: " + str(len(self.train_documents)))

    def keyword_extraction(self, dataset_name: str):
        super().keyword_extraction(dataset_name)

        # 训练先验知识
        model = kex.SingleTPR()
        model.train(self.train_documents, export_directory='./tmp/single_tpr')

        # 加载先验知识模型，提取关键词
        model.load('./tmp/single_tpr')
        time1 = time.time()

        json_line, _ = kex.get_benchmark_dataset(dataset_name)
        size = len(json_line)
        test_size = int(size / 5)
        json_line = json_line[:test_size]
        for line in json_line:
            text = line['source']
            results = model.get_keywords(text, n_keywords=3)
            predict_keywords = []
            # 构建结果
            for result in results:
                predict_keywords.append(result['stemmed'])
            self.output_list[";".join(line['keywords'])] = predict_keywords

        time2 = time.time()
        self.cost = int(time2 - time1)

    def download_data(self):
        pass


if __name__ == '__main__':
    single_tpr_rank_model = SingleTPR()
    single_tpr_rank_model.keyword_extraction("Krapivin2009")
    single_tpr_rank_model.compute_metric()
    # single_tpr_rank_model.show_output_list()
    print("single_tpr_rank_model.precision: {}".format(single_tpr_rank_model.precision))
    print("single_tpr_rank_model.recall: {}".format(single_tpr_rank_model.recall))
    print("single_tpr_rank_model.f_score: {}".format(single_tpr_rank_model.f_score))
