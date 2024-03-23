import unittest

from biz.config import read_config


class TestReadConfig(unittest.TestCase):
    def test_read_config(self):
        tuple1 = read_config()
        print(tuple1[0])
        print(type(tuple1[0]))
        print(tuple1[1])
        print(type(tuple1[1]))
