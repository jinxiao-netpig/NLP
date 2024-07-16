class Configuration:
    def __init__(self, n, k, num_iterations, d=3, delta=0.25, ce=0.25, precision=1000):
        """
        初始化 Vivaldi 坐标系统所需的基本参数

        :param n: 节点数量
        :param k: 邻居数量
        :param num_iterations: 调整坐标的迭代次数（目前还不确定是迭代因子还是边权重）
        :param d: 坐标的维度
        :param delta: 尺度因子（目前还不知道是做什么的）
        :param ce: 误差估计中的精度权重
        :param precision: 相对误差的精度
        :return:
        """

        self.num_nodes = n
        self.num_neighbors = k
        self.num_iterations = num_iterations
        self.num_dimension = d
        self.delta = delta
        self.ce = ce
        self.precision = precision

    def getNumInterations(self):
        return self.num_iterations

    def getNumNodes(self):
        return self.num_nodes

    def getNumNeighbors(self):
        return self.num_neighbors

    def getNumDimension(self):
        return self.num_dimension

    def getDelta(self):
        return self.delta

    def getCe(self):
        return self.ce

    def getPrecision(self):
        return self.precision
