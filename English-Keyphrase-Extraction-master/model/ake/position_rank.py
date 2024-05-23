import time

import kex

from model.ake.meta_method import MetaMethod


class PositionRank(MetaMethod):
    def __init__(self):
        pass

    def keyword_extraction(self, dataset_name: str):
        time1 = time.time()
        self.load_stopwords()
        self.filter_documents()

        model = kex.PositionRank()
        # 测试集取前500条数据
        size = 500
        if size > len(kex.get_benchmark_dataset(dataset_name)):
            size = len(kex.get_benchmark_dataset(dataset_name))
        json_line, _ = kex.get_benchmark_dataset(dataset_name)[:size]
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

    def computer_metric(self):
        pass

    def download_data(self):
        pass


if __name__ == '__main__':
    position_rank_model = PositionRank()
    position_rank_model.keyword_extraction("Inspec")
    position_rank_model.show_output_list()
