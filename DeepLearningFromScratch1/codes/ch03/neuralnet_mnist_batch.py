# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def cal_params_count(params):
    all_count = 0
    for i in range(len(params)):
        if (len(params[i]) == 1):
            all_count += params[i][0]
            continue
        row = params[i][0]
        col = params[i][1]
        all_count += row*col
    
    print("总参数量为：", all_count)

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

        w1, w2, w3 = network['W1'], network['W2'], network['W3']
        b1, b2, b3 = network['b1'], network['b2'], network['b3']

        print(w1.shape, w2.shape, w3.shape,)
        print(b1.shape, b2.shape, b3.shape,)

        all_params = w1.shape[0]*w1.shape[1] + w2.shape[0]*w2.shape[1] + w3.shape[0]*w3.shape[1] + b1.shape[0] + b2.shape[0] + b3.shape[0]

        print('最简单的2层神经网络的参数总量: ', all_params)
        # output: (784, 50) (50, 100) (100, 10)
        # (50,) (100,) (10,)
        # 最简单的2层神经网络的参数总量:  45360
        cal_params_count([(784, 50), (50,), (50, 100),(100,), (100, 10), (10,)])
    return network


def predict(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()

batch_size = 100 # 批数量
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
# output: Accuracy:0.9352
