'''
# 实现线性回归算法
'''

import numpy as np

from utils.features import prepare_for_training

class LinearRegression:
    """
    1. 对数据进行预处理操作
    2. 先得到所有的特征个数
    3. 初始化参数矩阵
    """
    def __init__(self, data, labels, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
        (data_processed,
         features_mean,
         features_deviation) = prepare_for_training(data, polynomial_degree, sinusoid_degree, normalize_data)

        self.data = data_processed                  # 所有样本数据
        self.labels = labels                        # 标签
        self.features_mean = features_mean          #
        self.features_deviation = features_deviation  # 特征的标准差
        self.polynomial_degree = polynomial_degree  # 多项式的特征变换次数
        self.sinusoid_degree = sinusoid_degree      # 正弦的次数
        self.normalize_data = normalize_data        # 是否归一化处理

        num_features = self.data.shape[1]           # 样本的特征数
        self.theta = np.zeros((num_features, 1))    # 初始化参数矩阵

    def train(self, alpha, num_iterations=500):
        """
        训练模块，执行梯度下降
        :param alpha:学习率
        """
        cost_history = self.gradient_descent(alpha, num_iterations)
        return self.theta, cost_history

    # 梯度下降算法
    def gradient_descent(self, alpha, num_iterations):
        """
        实际迭代模块，会迭代 num_iterations 次
        """
        cost_history = []       # 损失值
        for _ in range(num_iterations):
            self.gradient_step(alpha)
            cost_history.append(self.cost_function(self.data, self.labels))
        return cost_history

    def gradient_step(self, alpha):
        """
        梯度下降参数更新计算方法，注意是矩阵运算
        """
        num_examples = self.data.shape[0]       # 样本数
        prediction = LinearRegression.hypothesis(self.data, self.theta)
        delta = prediction - self.labels    # 预测值与正确值的差
        theta = self.theta
        theta = theta - alpha * (1/num_examples) * (np.dot(delta.T, self.data)).T
        self.theta = theta

    def cost_function(self, data, labels):
        """
        损失函数计算损失值，均方误差
        """
        num_examples = data.shape[0]
        delta = LinearRegression.hypothesis(self.data, self.theta) - labels   # [124,1]
        cost = (1/2) * np.dot(delta.T, delta) / num_examples    # [1,1] eg [[100]]
        return cost[0][0]

    # 模型预测
    @staticmethod
    def hypothesis(data, theta):
        predictions = np.dot(data, theta)
        return predictions

    # 得到当前的损失值,用于测试集
    def get_cost(self, data, labels):
        data_processed = prepare_for_training(data,
                             self.polynomial_degree,
                             self.sinusoid_degree,
                             self.normalize_data
                             )[0]

        return self.cost_function(data_processed, labels)

    def predict(self, data):
        """
        用训练好的参数模型，去预测得到回归值结果
        """
        data_processed = prepare_for_training(data,
                                              self.polynomial_degree,
                                              self.sinusoid_degree,
                                              self.normalize_data
                                              )[0]

        predictions = LinearRegression.hypothesis(data_processed, self.theta)
        return predictions



