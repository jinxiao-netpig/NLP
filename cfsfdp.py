import numpy as np


class CFSFDP:
    pattern = [
        "euclidean_distance",
    ]

    def __init__(self, epsilon, threshold):
        self.epsilon = epsilon  # 距离阈值，用于确定某点的局部密度
        self.threshold = threshold  # 密度峰值阈值
        pass

    def fit(self):
        pass

    def build_local_density_list(self):
        pass

    def build_relative_density_list(self):
        pass

    def get_distance(self):
        pass

    def compute_distance(self, x: np.ndarray[float], y: np.ndarray[float], distance_pattern: str = "euclidean_distance") -> float:
        """
        计算两个数据点之间的距离

        :param distance_pattern: 距离类型。默认欧氏距离
        :param x: x 点的坐标
        :param y: y 点的坐标
        :return: x 和 y 之间的距离
        """

        # 检查两点的坐标列表长度是否相等
        if x.size != y.size:
            raise Exception("两点坐标列表长度不同")
        # 检查距离类型输入
        if distance_pattern not in self.pattern:
            raise Exception("不存在这种距离计算模式")

        # 根据 pattern 使用对应的函数
        if distance_pattern == "euclidean_distance":
            return self.compute_euclidean_distance(x, y)

    @staticmethod
    def compute_euclidean_distance(x: np.ndarray[float], y: np.ndarray[float]) -> float:
        """
        计算两个数据点之间的欧式距离

        :param x: x 点的坐标
        :param y: y 点的坐标
        :return: x 和 y 之间的欧氏距离
        """

        distance = (np.sum(np.square(np.subtract(x, y)))) ** 0.5

        return distance

    def build_distance_matrix(self):
        pass

    def build_density_peaks_list(self):
        pass
