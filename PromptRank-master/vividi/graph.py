from typing import List


class Graph:
    """
    :param N: 节点数量
    :param adjacentList: 邻接表，一个词典，键是节点(v)，值是一个元组列表(w, RTTuw)，其中 w 是另一个节点，RTTuw 是两个节点之间的边权重
    """

    def __init__(self, n):
        """
        :param n: 节点数量
        :return:
        """

        self.N = n
        self.adjacentList = {}

    def addVertex(self, v, w, rtt):
        """
        添加节点以及与其它节点之间的边

        :param v: 节点 v
        :param w: 节点 w
        :param rtt: 两点之间的边权重
        :return:
        """

        assert rtt >= 0  # 边权重不能为负值
        if v == w:
            return
        if v not in self.adjacentList.keys():
            self.adjacentList[v] = [(w, rtt)]
        else:
            self.adjacentList[v].append((w, rtt))

    def getAdjacent(self, v) -> List[tuple]:
        """
        查找节点 v 的邻接关系

        :param v: 节点 v
        :return: 节点 v 对应的邻接节点的元组列表
        """

        return self.adjacentList[v]

    def getRTT(self, v, w):
        """
        查找两节点之间的边权重

        :param v: 节点 v
        :param w: 节点 w
        :return: 节点 v 和 w 之间的边权重
        """

        if v == w:
            return 0
        for z, rtt in self.adjacentList[v]:
            if z == w:
                return rtt
        return None

    def getGraphSize(self):
        """
        获得图的大小，即节点数

        :return: 图的节点数
        """

        return self.N

    def getAdjacentList(self):
        """
        获得邻接表

        :return: 邻接表
        """

        return self.adjacentList
