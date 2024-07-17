import random
import sys
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
        # 将传入的图（graph）和配置（configuration）保存为实例变量，并获取配置中的维度数（d）
        self.graph = graph
        self.configuration = configuration
        self.d = configuration.getNumDimension()

        # 初始化节点的位置为全零数组，初始化每个节点的误差为1，并初始化误差历史和移动长度历史的数组
        self.positions = [[0] * self.d for _ in range(self.configuration.getNumNodes())]
        self.errors = [1] * self.configuration.getNumNodes()
        self.error_history = [0] * self.configuration.getNumInterations()
        self.move_len_history = [0] * self.configuration.getNumInterations()

    def _random_direction(self):
        """
        随机方向生成

        :return: 生成一个在每个维度上取值为0到10的随机方向向量
        """

        return [random.randint(0, 10) for _ in range(self.d)]

    @staticmethod
    def _unit_vector(self, vector):
        """
        计算一个向量的单位向量。

        :param vector: 一个向量
        :return: 如果向量全为零，则返回该向量，否则返回归一化后的向量。
        """

        if all(vector) == 0:
            return vector
        return vdiv(vector, norm(vector))

    @staticmethod
    def _update_progress(self, progress):
        """
        更新和显示算法运行进度的文本进度条。
        """

        length = 50
        done = int(round(progress * length))
        left = length - done

        sys.stdout.write("\r[" + "=" * done + ">" + " " * left + "] " + str(int(100 * progress)) + " % ")
        sys.stdout.flush()

    @staticmethod
    def _clear_progress(self):
        """
        进度清除
        """

        sys.stdout.write("\r")
        sys.stdout.flush()
        print()

    def run(self):
        """
        Vivaldi算法核心
        """

        # 获得迭代数和误差估计中的精度权重
        iters = self.configuration.getNumInterations()
        ce = self.configuration.getCe()

        # 开始迭代
        for i in range(iters):
            # rtt_prediction = self.getRTTGraph()

            temp_error_history = 0.0
            # 为每个节点随机挑选 K 个邻居
            for node, neighbors in self.graph.getAdjacentList().items():
                random_neighbors = [random.choice(neighbors) for _ in range(self.configuration.getNumNeighbors())]

                error_sum = 0
                move_len = 0
                for neighbor, rtt_measured in random_neighbors:
                    remote_confidence = self.errors[node] / (self.errors[node] + self.errors[neighbor])

                    absolute_error = (self.distance(node, neighbor) - rtt_measured)
                    relative_error = abs(absolute_error) / rtt_measured

                    error_sum += (relative_error * ce * remote_confidence) + (
                            self.errors[node] * (1 - ce * remote_confidence))
                    temp_error_history += abs(absolute_error)

                    # 一开始所有节点位置都相同，所以给个初始的随机位置
                    direction = vsub(self.positions[neighbor], self.positions[node])
                    if not all(direction):
                        direction = self._random_direction()
                    direction = self._unit_vector(direction)

                    delta = ce * remote_confidence

                    # 检查节点在RTT方面需要向/远离邻居“移动”多少，然后按照Vivaldi算法计算新的坐标
                    movement = vmul(direction, delta * absolute_error)
                    move_len += norm(movement)
                    self.positions[node] = vadd(self.positions[node], movement)

                self.errors[node] = error_sum / len(random_neighbors)
                self.move_len_history[i] += move_len / (self.configuration.getNumNodes() * len(random_neighbors))

            self.error_history[i] = (
                    temp_error_history / (self.configuration.getNumNodes() * self.configuration.getNumNeighbors()))
            # self._update_progress(float(i)/iters)
        # self._clear_progress()

        # pyplot.plot(range(len(errorplot)), errorplot)
        # pyplot.ylim(ymin=0)
        # pyplot.show()

    def getRTTGraph(self):
        """
        根据当前节点位置预测RTT图

        :return: 二维矩阵，节点 id -- 邻居节点 id，值是节点之间的边权重
        """

        prediction = [0] * self.configuration.getNumNodes()

        for nid, neighbors in self.graph.getAdjacentList().items():
            prediction[nid] = [0] * self.configuration.getNumNodes()
            for (neighbor, rtt) in neighbors:
                prediction[nid][neighbor] = norm(vsub(self.positions[nid], self.positions[neighbor]))

        return prediction

    def distance(self, fr, to):
        """
        计算两节点之间的欧几里得距离

        :param fr: 起始点
        :param to: 目标点
        :return: 两节点之间的欧几里得距离
        """

        return norm(vsub(self.positions[fr], self.positions[to]))

    def getPositions(self, node):
        """
        获取节点位置

        :param node: 节点
        :return: 节点坐标
        """

        return self.positions[node]

    def getRelativeError(self, predicted_graph):
        """
        相对误差计算

        :param predicted_graph: 预测RTT图
        :return: 预测RTT图与实际RTT图的相对误差
        """

        rerr = []
        i = 0
        for neighbors in predicted_graph:
            r = 0
            j = 0
            for rtt_predicted in neighbors:
                rtt_measured = self.graph.getRTT(i, j)
                if rtt_measured != 0:
                    r += abs((rtt_predicted - rtt_measured) / rtt_measured)
                j += 1
            rerr.append(r / j)
            i += 1
        return rerr

    def computeCDF(self, input_):
        """
        计算输入数据的累积分布函数（CDF）
        """

        x = sorted(input_)
        y = map(lambda x: x / float((len(input_) + 1)), range(len(input_)))
        return x, y
