import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from twolayernet import *
from common.optimizer import *

# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000  # 繰り返しの回数を適宜設定する
train_size = x_train.shape[0]
batch_size = 200
learning_rate = 0.1

train_loss_list = []
test_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)
optimizer = Nesterov(lr=0.1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 勾配の計算
    grad = network.gradient(x_batch, t_batch)

    # パラメータの更新

    optimizer.update(network.params, grad)

    if i % iter_per_epoch == 0:
        train_loss_list.append(network.loss(x_train, t_train))
        test_loss_list.append(network.loss(x_test, t_test))
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# グラフの描画

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
x = np.arange(len(train_loss_list))
ax1.plot(x, train_loss_list, label='train loss', marker='o')
ax1.plot(x, test_loss_list, label='test loss', marker='s')
# ax1.set_xlabel("epochs")
ax1.set_ylabel("loss")
ax1.figure.legend(loc='upper right')

ax2 = fig.add_subplot(2, 1, 2)
x = np.arange(len(train_acc_list))
ax2.plot(x, train_acc_list, label='train acc', marker='o')
ax2.plot(x, test_acc_list, label='test acc', marker='s')
ax2.set_xlabel("epochs")
ax2.set_ylabel("accuracy")
ax2.set_ylim(0, 1)
ax2.figure.legend(['train acc', 'test acc'], loc='lower right')
plt.show()
