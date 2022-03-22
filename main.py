# -*- encoding: utf-8 -*-
'''
Filename         :main.py
Description      :利用2层全连接网络对mnist数据集进行分类
Time             :2022/03/19 22:46:59
Author           :管鸿鑫
Version          :1.0
'''

import numpy as np
from read_mnist import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.special

def convert_to_one_hot(labels_orig, class_nums):
    """将0-9的标签值转换成one-hot码形式

    Args:
        labels_orig (_type_): 原始0-9标签
        class_nums (_type_): 类别数

    Returns:
        labels: one-hot形式的label, shape(类别数, 样本数)
    """
    labels = np.zeros((class_nums,labels_orig.size))

    for i in range(labels_orig.size):
        labels[int(labels_orig[i]),i] = 1
    
    return labels

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.lr = learning_rate

        self.w_ih = np.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.hidden_nodes, self.input_nodes))
        self.b_h = np.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.hidden_nodes, 1))
        self.w_ho = np.random.normal(0.0, pow(self.output_nodes, -0.5), (self.output_nodes, self.hidden_nodes))
        self.b_o = np.random.normal(0.0, pow(self.output_nodes, -0.5), (self.output_nodes, 1))

        self.activation_function = lambda x: scipy.special.expit(x)

        self.cost = 0
    
    def train(self, X, Y):

        X = X.reshape(784,-1)
        Y = Y.reshape(10,-1)
        m = X.shape[1]

        # 前向传播
        # ---隐含层
        Z1 = np.dot(self.w_ih, X) + self.b_h
        A1 = self.activation_function(Z1)
        # ---输出层
        Z2 = np.dot(self.w_ho, A1) + self.b_o
        A2 = self.activation_function(Z2)

        # 计算损失
        self.cost = self.comput_cost(A2, Y)

        # 计算梯度
        output_errors = Y - A2
        hidden_errors = np.dot(self.w_ho.T, output_errors)

        # BP算法
        self.w_ho += self.lr * 1 / m * np.dot((output_errors * A2 * (1.0 - A2)), np.transpose(A1))
        self.b_o += self.lr * 1 / m * np.sum(output_errors * A2 * (1.0 - A2), axis = 1, keepdims=True)
        self.w_ih += self.lr * 1 / m * np.dot((hidden_errors * A1 * (1.0 - A1)), np.transpose(X))
        self.b_h += self.lr * 1 / m * np.sum(hidden_errors * A1 * (1.0 - A1), axis = 1, keepdims=True)
    
    def predict(self, X):

        X = X.reshape(784,-1)

        Z1 = np.dot(self.w_ih, X) + self.b_h
        A1 = self.activation_function(Z1)

        Z2 = np.dot(self.w_ho, A1)
        A2 = self.activation_function(Z2)

        predi = np.argmax(A2, axis=0)
        return predi
    
    def comput_cost(self, AL, Y):
        m = Y.shape[1]
        cost = 1/m*1/2* np.sum((AL-Y)**2)

        return cost

# 读取数据集
train_images = load_train_images()          #(784=28*28, 60000)
train_labels_orig = load_train_labels()     #(60000,)
test_images = load_test_images()            #(784=28*28, 10000)
test_labels_orig = load_test_labels()       #(10000,)

# 将labels转换成one-hot编码
num_classes = 10
train_labels = convert_to_one_hot(train_labels_orig, num_classes)
test_labels = convert_to_one_hot(test_labels_orig, num_classes)

# 查看前10张看是否标签与图片对应、one-hot编码是否正确，关闭figure窗口才会继续运行
# for i in range(10):
#     print('原始标签为:',train_labels_orig[i])
#     print('one-hot标签为:',train_labels[:,i])
#     plt.imshow(train_images[:,i].reshape(28,28),cmap='gray')
#     plt.show()

# 归一化
train_images = train_images/255
test_images = test_images/255



input_nodes = 784  
hidden_nodes = 200  
output_nodes = 10  
learning_rate = 0.1

n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# 训练(如果batch_size设得越大，epoch也要设置多一点才可以下降到比较小的loss，得到比较高的准确率。也不知道原理......)
epochs = 10
batch_size = 10
for e in range(epochs):
    print('Epoch %d/%d' % (e, epochs))
    for i in tqdm(range(0, 60000, batch_size)):
        X = train_images[:, i:i+batch_size]
        Y = train_labels[:, i:i+batch_size]
        n.train(X, Y)
    print("cost is ", n.cost)
    # 每次epoch随机打乱
    per = np.random.permutation(train_images.shape[1])
    train_images = train_images[:, per]		#获取打乱后的训练数据
    train_labels = train_labels[:, per]
    train_labels_orig = train_labels_orig[per]

# 保存参数
parameters = {"w_ih": n.w_ih,
              "b_h": n.b_h,
              "W_ho": n.w_ho,
              "b_o": n.b_o}

np.save('parameters.npy', parameters)

# 加载参数
# parameters = np.load('parameters.npy', allow_pickle='TRUE')
# n.w_ih = parameters.item()['w_ih']
# n.b_h = parameters.item()['b_h']
# n.w_ho = parameters.item()['W_ho']
# n.b_o = parameters.item()['b_o']

# 预测得到非one-hot label
train_pre_ = n.predict(train_images)
test_pre_ = n.predict(test_images)

# 统计预测值与真实标签不同的个数
train_pre = train_pre_==train_labels_orig
test_pre = test_pre_==test_labels_orig
train_right = np.sum(train_pre)
test_right = np.sum(test_pre)

# 找出测试集合预测错误的索引
test_wrong = np.where(test_pre==0)[0]

# 计算预测准确率
train_accu = train_right/60000
test_accu = test_right/10000

print(f'训练集准确率为：{round(train_accu * 100, 2)}%')
print(f'测试集准确率为：{round(test_accu * 100, 2)}%')

# 可视化
# 画出测试集的前10张
fig, axes = plt.subplots(figsize=(8, 8), nrows = 5, ncols = 5)  
i = 0
for row in range(5):  
    for col in range(5):

        axes[row][col].imshow(test_images[:,i].reshape(28,28), cmap = 'Greys', interpolation = 'None')
        axes[row][col].set_title(f'predict:{test_pre_[i]}')
        axes[row][col].set_xticks([])
        axes[row][col].set_yticks([])

        i += 1
plt.show()

# 画出测试集判错的前10张
fig, axes = plt.subplots(figsize=(8, 8), nrows = 5, ncols = 5)  
i = 0
for row in range(5):  
    for col in range(5):

        axes[row][col].imshow(test_images[:,test_wrong[i]].reshape(28,28), cmap = 'Greys', interpolation = 'None')
        axes[row][col].set_title(f'label:{test_labels_orig[test_wrong[i]]}, predict:{test_pre_[test_wrong[i]]}')
        axes[row][col].set_xticks([])
        axes[row][col].set_yticks([])

        i += 1
plt.show()
