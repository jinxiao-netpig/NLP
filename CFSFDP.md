# CFSFDP 聚类算法

Constrained Fast Search for Density Peaks，一种聚类算法，基于密度峰值原理和约束条件识别数据集中的聚类结构。

# 步骤

1. **密度峰值原理：**
   密度峰值原理是CFSFDP算法的核心概念，它认为聚类的中心点应该是在数据集中局部密度较高的区域，并且周围的点密度相对较低。密度峰值原理通过计算每个数据点的局部密度和相对密度来判断其是否为聚类的中心点。
2. **数据点的局部密度（Local Density）：**
   对于每个数据点，可以通过计算其与其他数据点之间的距离来得到一个有序的距离列表。局部密度可以定义为在给定距离阈值ε内与数据点距离小于等于ε的数据点数量。局部密度越高，表示数据点周围的点越多，可能是聚类的中心点。
3. **数据点的相对密度（Relative Density）：**
   相对密度可以定义为在给定距离阈值ε内比数据点的局部密度更高或相等的数据点数量。相对密度越高，表示数据点的局部密度相对较高，可能是聚类的中心点。
4. **密度峰值（Density Peaks）：**
   密度峰值可以定义为局部密度和相对密度的乘积。通过计算每个数据点的密度峰值，可以找到潜在的聚类中心点。
5. **聚类中心点的选择：**
   CFSFDP算法选择密度峰值大于某个阈值的数据点作为聚类的中心点。阈值的选择可以根据具体问题来确定。
6. **距离计算：**
   CFSFDP算法使用欧氏距离作为数据点之间的距离度量。
7. **约束条件：**
   CFSFDP算法还引入了约束条件，以限制聚类结果。约束条件可以是已知的先验信息，例如两个数据点属于同一个聚类或不属于同一个聚类等。
8. **聚类分配：**
   CFSFDP算法将每个数据点分配到最近的聚类中心点，形成最终的聚类结果。

# 代码（GPT版）

```python
import numpy as np
from scipy.spatial.distance import euclidean

class CFSFDP:
    def __init__(self, epsilon, min_pts, threshold):
        self.epsilon = epsilon  # 距离阈值
        self.min_pts = min_pts  # 最小邻域数
        self.threshold = threshold  # 密度峰值阈值
        self.labels = None  # 聚类标签

    def fit(self, X):
        n = X.shape[0]  # 数据点数量
        distances = np.zeros((n, n))  # 存储数据点之间的距离

        # 计算数据点之间的距离
        for i in range(n):
            for j in range(i+1, n):
                distances[i][j] = euclidean(X[i], X[j])
                distances[j][i] = distances[i][j]

        # 计算每个数据点的局部密度
        density = np.zeros(n)
        for i in range(n):
            density[i] = np.sum(distances[i] <= self.epsilon)

        # 计算每个数据点的相对密度
        relative_density = np.zeros(n)
        for i in range(n):
            relative_density[i] = np.sum(density > density[i])

        # 计算每个数据点的密度峰值
        density_peak = density * relative_density

        # 筛选出密度峰值大于阈值的数据点作为聚类中心
        center_indices = np.where(density_peak >= self.threshold)[0]

        # 初始化聚类标签
        self.labels = np.zeros(n, dtype=int)

        # 分配数据点到聚类中心
        cluster_id = 1
        for center_index in center_indices:
            if self.labels[center_index] == 0:
                self.labels[center_index] = cluster_id

            for i in range(n):
                if distances[center_index][i] <= self.epsilon and self.labels[i] == 0:
                    self.labels[i] = cluster_id

            cluster_id += 1

        return self.labels

```

1. 初始化：在`__init__`方法中，传入距离阈值epsilon、最小邻域数min_pts和密度峰值阈值threshold，并将聚类标签labels初始化为None。
2. 计算距离矩阵：使用`scipy.spatial.distance`库中的`euclidean`函数计算数据点之间的欧氏距离，存储在distances矩阵中。
3. 计算局部密度：遍历每个数据点，计算其在距离阈值epsilon内的邻域数，即与其距离小于等于epsilon的数据点数量，存储在density数组中。
4. 计算相对密度：遍历每个数据点，计算其相对于其他数据点的密度高于等于自身的数据点数量，存储在relative_density数组中。
5. 计算密度峰值：将局部密度和相对密度相乘，得到每个数据点的密度峰值，存储在density_peak数组中。
6. 筛选聚类中心：根据密度峰值阈值threshold，筛选出密度峰值大于等于阈值的数据点作为聚类中心，存储在center_indices数组中。
7. 分配聚类标签：遍历每个聚类中心，将其标记为当前聚类id，然后将距离其距离小于等于epsilon的未分配数据点也标记为当前聚类id，最后聚类id递增。
8. 返回聚类结果：将聚类标签labels返回作为聚类结果。

# 总结

CFSFDP 聚类算法具有较快的计算速度和较好的聚类效果，在处理大规模数据集和带约束的聚类问题时具有优势。