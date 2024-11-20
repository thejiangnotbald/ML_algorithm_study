"""
单特征线性回归的梯度下降，只有一个特征，一个标签
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from linear_regression import LinearRegression

data = pd.read_csv('../data/world-happiness-report-2017.csv')

# 划分得到训练集和测试集
train_data = data.sample(frac=0.8)    # 从data中随机抽取80%作为训练集
test_data = data.drop(train_data.index)

input_param_name = 'Economy..GDP.per.Capita.'
output_param_name = 'Happiness.Score'

# x_train必须是二维数组，此时shape：[123,1]
x_train = train_data[[input_param_name]].values
y_train = train_data[[output_param_name]].values

x_test = test_data[[input_param_name]].values
y_test = test_data[[output_param_name]].values

plt.scatter(x_train, y_train, label='Train data')
plt.scatter(x_test, y_test, label='Test data')
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title('Happy')
plt.legend()        # 显示每个数据集的标签 label
plt.show()

# 训练线性回归模型
num_iterations = 500
learning_rate = 0.01
linear_regression = LinearRegression(x_train, y_train)
(theta, cost_history) = linear_regression.train(learning_rate, num_iterations)

# 可视化
print('开始的损失：', cost_history[0])
print('训练后的损失：', cost_history[-1])

plt.plot(range(num_iterations), cost_history)
plt.xlabel('Iterations')
plt.ylabel('cost')
plt.title('Gradient Descent Progress')
plt.show()

#  测试模型
predictions_num = 100
x_predictions = np.linspace(x_test.min(), x_train.max(), predictions_num).reshape(predictions_num, 1)
y_predictions = linear_regression.predict(x_predictions)

plt.scatter(x_train, y_train, label='Train data')
plt.scatter(x_test, y_test, label='Test data')
plt.plot(x_predictions, y_predictions, label='Predictions')
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title('Happy')
plt.legend()        # 显示每个数据集的标签 label
plt.show()
