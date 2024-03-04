from typing import TypeVar

import numpy as np

T = TypeVar('T')


class CFSFDP:
    pattern = [
        "euclidean_distance",
    ]
    local_density_list = {}
    relative_density_list = {}
    density_peaks_list = {}

    def __init__(self, epsilon: float, threshold: float, points: dict[T, np.ndarray]):
        self.epsilon = epsilon  # 距离阈值。用于确定某点的局部密度
        self.threshold = threshold  # 密度峰值阈值。用于确定聚类中心的个数
        self.points = points  # 所有数据点的坐标集合
        pass

    def fit(self):
        pass

    def build_local_density_list(self):
        """
        构建数据点-局部密度列表

        :return:
        """

        for key in self.points.keys():
            self.local_density_list[key] = self.get_point_local_density(key)

    def build_relative_density_list(self):
        """
        构建数据点-相对密度列表

        :return:
        """

        for key in self.points.keys():
            self.relative_density_list[key] = self.get_point_relative_density(key)

    def build_density_peaks_list(self):
        """
        构建数据点-密度峰值列表

        :return:
        """

        for key in self.points.keys():
            density_peak = self.local_density_list[key] * self.relative_density_list[key]
            self.density_peaks_list[key] = density_peak

    def get_point_relative_density(self, point: T) -> int:
        """
        计算 p 点的相对密度

        :param point: p 点名称
        :return: p 点的相对密度
        """

        relative_density = 0
        point_ld = self.local_density_list[point]

        for ld in self.local_density_list.values():
            if ld >= point_ld:
                relative_density += 1

        return relative_density

    def __get_distance(self, x: np.ndarray[float], y: np.ndarray[float],
                       distance_pattern: str = "euclidean_distance") -> float:
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
            return self.__get_euclidean_distance(x, y)

    @staticmethod
    def __get_euclidean_distance(x: np.ndarray[float], y: np.ndarray[float]) -> float:
        """
        计算两个数据点之间的欧式距离

        :param x: x 点的坐标
        :param y: y 点的坐标
        :return: x 和 y 之间的欧氏距离
        """

        distance = (np.sum(np.square(np.subtract(x, y)))) ** 0.5

        return distance

    def __build_distance_list(self, point: T) -> list[tuple[T, float]]:
        """
        构建某个点的距离列表

        :param point: 点名称。用于确定点的坐标
        :return: 距离列表
        """

        res = []
        point_loc = self.points[point]

        for p, loc in self.points.items():
            res.append((p, self.__get_distance(point_loc, loc)))

        return res

    def __get_point_local_density(self, distance_list: list[tuple[T, float]]) -> int:
        """
        根据某点的距离列表计算局部密度

        :param distance_list: 某点的距离列表
        :return: 局部密度
        """

        local_density = 0

        for item in distance_list:
            distance = item[1]

            if distance <= self.epsilon:
                local_density += 1

        return local_density

    def get_point_local_density(self, point: T) -> int:
        """
        计算 p 点的局部密度

        :param point: p 点名称
        :return: p 点的局部密度
        """

        distance_list = self.__build_distance_list(point)
        local_density = self.__get_point_local_density(distance_list)

        return local_density
