import time

import kex

from model.ake.meta_method import MetaMethod


class PositionRank(MetaMethod):
    def __init__(self):
        pass

    def keyword_extraction(self, dataset_name: str):
        super().keyword_extraction(dataset_name)
        time1 = time.time()

        model = kex.PositionRank()
        # 测试集取前500条数据
        size = 500
        json_line, _ = kex.get_benchmark_dataset(dataset_name)
        if size > len(json_line):
            size = len(json_line)

        json_line = json_line[:size]
        for line in json_line:
            text = line['source']
            results = model.get_keywords(text, n_keywords=5)
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
    position_rank_model = PositionRank()
    position_rank_model.keyword_extraction("Krapivin2009")
    # position_rank_model.show_output_list()
    position_rank_model.compute_metric()
    print("single_tpr_rank_model.precision: {}".format(position_rank_model.precision))
    print("single_tpr_rank_model.recall: {}".format(position_rank_model.recall))
    print("single_tpr_rank_model.f_score: {}".format(position_rank_model.f_score))
