# 构建softmax函数
from dezero.models import MLP
from dezero import Variable, as_variable
import dezero.functions as F
import numpy as np
model = MLP((10, 3))
# x = np.array([[0.2, -0.4]])
# y = model(x)
# print(y)

# def softmax1d(x):
#     x = as_variable(x)
#     y = F.exp(x)
#     sum_y = F.sum(y)
#     return y / sum_y

x = np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]])
t =  np.array([2, 0, 1, 0])
y = model(x)
loss = F.softmax_cross_entropy(y, t)
print(loss)