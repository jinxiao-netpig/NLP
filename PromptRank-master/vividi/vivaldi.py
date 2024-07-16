from math import sqrt

from vividi.configuration import Configuration
from vividi.graph import Graph


def vadd(a, b):
    """
    向量相加

    :param a: 向量 a
    :param b: 向量 b
    :return: 一个新向量，每个元素是两向量相应元素的和
    """

    return [a[i] + b[i] for i in range(len(a))]


def vsub(a, b):
    """
    向量相减

    :param a: 向量 a
    :param b: 向量 b
    :return: 一个新向量，每个元素是两向量相应元素的差（a-b）
    """

    return [a[i] - b[i] for i in range(len(a))]


def vmul(v, factor):
    """
    向量与标量相乘

    :param v: 向量 v
    :param factor: 标量 factor
    :return: 一个新向量，每个元素是 v 的相应元素乘以 factor
    """

    return [factor * a for a in v]


def vdiv(v, divisor):
    """
    向量与标量相除

    :param v: 向量 v
    :param divisor: 标量 divisor
    :return: 一个新向量，每个元素是 v 的相应元素除以 divisor
    """

    return [a / divisor for a in v]


def norm(v):
    """
    向量的范数（长度）

    :param v: 向量 v
    :return: 向量 v 的长度，计算方法是所有元素平方和的平方根
    """

    return sqrt(sum([a * a for a in v]))


class Vivaldi:
    def __init__(self, graph: Graph, configuration: Configuration):
        self.graph = graph
        self.configuration = configuration
        self.d = configuration.getNumDimension()
