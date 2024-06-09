import sys
from typing import TypeVar

import matplotlib.pyplot as plt
import numpy as np

# 2014年的文章，Science
# 将其应用在 NLP 是2016年的文章
# 我这边改进是变成监督式，密度峰值阈值才去乘法

T = TypeVar('T')


class CFSFDP:
    pattern = [
        "euclidean_distance",
        "cosine_similarity"
    ]
    local_density_list: dict[T, int] = {}  # 数据点:局部密度
    relative_density_list = {}  # 数据点:相对密度（最小距离）
    density_peaks_list = {}  # 数据点:密度峰值
    center_indices_list = {}  # 数据点:聚类中心id

    def __init__(self, epsilon: float, threshold: float, points: dict[T, np.ndarray]):
        self.epsilon = epsilon  # 距离阈值。用于确定某点的局部密度
        self.threshold = threshold  # 密度峰值阈值。用于确定聚类中心的个数
        self.points = points  # 所有数据点的坐标集合
        pass

    def set_epsilon(self, epsilon: float):
        self.epsilon = epsilon

    def set_threshold(self, threshold: float):
        self.threshold = threshold

    def fit(self):
        self.build_local_density_list()  # 计算局部密度
        self.build_relative_density_list()  # 计算相对密度
        self.build_density_peaks_list()  # 计算密度峰值
        self.build_center_indices_list()  # 分配聚类标签

    def build_local_density_list(self):
        """
        构建 数据点-局部密度 列表

        :return:
        """

        for key in self.points.keys():
            self.local_density_list[key] = self.get_point_local_density(key)

    def build_relative_density_list(self):
        """
        构建 数据点-相对密度 列表

        :return:
        """

        for key in self.points.keys():
            self.relative_density_list[key] = self.get_point_relative_density(key)

    def build_density_peaks_list(self):
        """
        构建 数据点-密度峰值 列表

        :return:
        """

        for key in self.points.keys():
            density_peak = self.local_density_list[key] * self.relative_density_list[key]
            self.density_peaks_list[key] = density_peak

    def build_center_indices_list(self):
        """
        构建 数据点-聚类中心id 列表

        :return:
        """

        cluster_id = 1  # id 值越大，密度峰值越小
        peaks_list = []

        # 获得聚类中心集合
        for point, peak in self.density_peaks_list.items():
            if peak >= self.threshold:
                peaks_list.append((point, peak))

        # 按照密度峰值大小，排序聚类中心集合

        def sort_key(point_peak: tuple[T, float]):
            return point_peak[1]

        # 降序排序
        peaks_list.sort(key=sort_key, reverse=True)

        # 给聚类中心赋予 id 值
        for item in peaks_list:
            self.center_indices_list[item[0]] = cluster_id
            cluster_id += 1

    def get_point_relative_density(self, point: T) -> float:
        """
        计算 p 点的相对密度

        :param point: p 点名称
        :return: p 点的相对密度
        """

        relative_density = sys.float_info.max
        point_ld = self.local_density_list[point]

        for point_Y, ld in self.local_density_list.items():
            # 如果是自己就跳过
            if point_Y == point:
                continue
            # 如果自己就是局部密度最大的点，那么相对密度直接也给最大值
            if ld > point_ld:
                relative_density = min(relative_density, self.__get_distance(self.points[point], self.points[point_Y]))

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
    def __get_cosine_similarity_distance(x: np.ndarray[float], y: np.ndarray[float]) -> float:
        """
        计算两个数据点之间的余弦相似度

        :param x: x 点的坐标
        :param y: y 点的坐标
        :return: x 和 y 之间的余弦相似度
        """
        pass

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

        return local_density - 1

    def get_point_local_density(self, point: T) -> int:
        """
        计算 p 点的局部密度

        :param point: p 点名称
        :return: p 点的局部密度
        """

        distance_list = self.__build_distance_list(point)
        local_density = self.__get_point_local_density(distance_list)

        return local_density

    def points_dict_to_slice(self) -> list[tuple]:
        """
        将 points 这个字典转成坐标的列表

        :return: 坐标的列表
        """

        res = []
        for value in self.points.values():
            res.append(tuple(value.tolist()))

        return res

    def centers_dict_to_slice(self) -> list[tuple]:
        """
        将 center_indices_list 这个字典转成坐标的列表

        :return: 坐标的列表
        """

        res = []
        centers = self.center_indices_list.keys()
        for center_point in centers:
            res.append(tuple(self.points[center_point].tolist()))

        return res


if __name__ == '__main__':
    epsilon = 0.5
    threshold = 4
    points = {
        "a": np.array([0, 0]),
        "b": np.array([0.1, 0]),
        "c": np.array([1, 0]),
        "d": np.array([0, 10]),
        "e": np.array([0, 4]),
        "f": np.array([1, 17]),
        "g": np.array([2.4, 1]),
        "h": np.array([0.7, 5]),
        "i": np.array([6, 5]),
        "j": np.array([10, 1]),
        "k": np.array([3, 1.3]),
        "l": np.array([7, 7.1]),
        "m": np.array([6, 7.1]),
        "n": np.array([5, 7.1]),
        "o": np.array([8, 7]),
        "p": np.array([3, 3.5]),
        "q": np.array([3.8, 7.1]),
        "r": np.array([3.5, 3]),
        "s": np.array([4, 7.1]),
        "t": np.array([6, 6]),
        "u": np.array([5, 5]),
        "v": np.array([8, 9]),
        "w": np.array([5.2, 2.5]),
    }

    model = CFSFDP(epsilon=epsilon, threshold=threshold, points=points)
    model.fit()

    # 打印
    print("local_density_list: {}".format(model.local_density_list))
    print("relative_density_list: {}".format(model.relative_density_list))
    print("density_peaks_list: {}".format(model.density_peaks_list))
    print("center_indices_list: {}".format(model.center_indices_list))

    # 画图
    plt.figure()
    data_points = model.points_dict_to_slice()
    x_coords, y_coords = zip(*data_points)
    plt.scatter(x_coords, y_coords, color="blue", marker="o")
    # 画出聚类中心
    center_points = model.centers_dict_to_slice()
    x, y = zip(*center_points)
    plt.scatter(x, y, color="red", marker="o")
    plt.title("CFSFDP Clustering Result")
    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")
    plt.show()
