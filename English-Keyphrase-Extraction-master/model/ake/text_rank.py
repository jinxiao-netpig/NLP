import kex

from model.ake.meta_method import MetaMethod


class TEXTRANK(MetaMethod):
    def __init__(self):
        pass

    def keyword_extraction(self):
        pass


if __name__ == '__main__':
    json_line, language = kex.get_benchmark_dataset('Inspec')
    print("json_line: ", json_line[0], len(json_line), type(json_line))
    print("language: ", language)
