"""
* @描述: 决策树例子
* @作者: 草莓夹心糖
* @创建时间: 2022/11/8 20:23
"""
import pandas as pd
import sys
import os
import numpy as np
import pandas as pd
from pre_process import processing, pre_pro_test   # 导入数据预处理的函数

try:
    data = pd.read_csv('titanic/train.csv')
except:
    print("请确保数据集存在, 且路径无误")
    sys.exit()
data = processing(data)   # 将数据进行预处理
data_train = data[0]
data_target = data[1]


# 搭建神经网络
import tensorflow as tf
x = tf.placeholder(dtype='float', shape=[None, 12])
y = tf.placeholder(dtype='float', shape=[None, 1])

# 前向传播过程
weight = tf.Variable(tf.random_normal([12,1]))
bias = tf.Variable(tf.random_normal([1]))
output = tf.matmul(x, weight) + bias
pred = tf.cast(tf.sigmoid(output) > 0.5,tf.float32)  # 预测结果大于0.5值设为1，否则为0
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output))  # 交叉熵损失函数
train_step = tf.train.GradientDescentOptimizer(0.0003).minimize(loss)  # 梯度下降法训练
accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, y), tf.float32))


# 开始训练
loss_train = []
train_acc = []
# 四、运行session(若模型不存在, 则建模)
if not os.path.exists('models/checkpoint'):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  # 初始化所有变量
        for i in range(25000):
            for n in range(len(data_target) // 100 + 1):
                batch_xs = data_train[n * 100:n * 100 + 100]
                batch_ys = data_target[n * 100:n * 100 + 100]
                sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
            if i % 5000 == 0:
                loss_temp = sess.run(loss, feed_dict={x: batch_xs, y: batch_ys})
                loss_train.append(loss_temp)
                train_acc_temp = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys})
                train_acc.append(train_acc_temp)
                print(loss_temp, train_acc_temp)
        print("训练结束")

        # 保存模型
        saver = tf.train.Saver()
        model_path = 'models/model.ckpt'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        save_path = saver.save(sess, model_path)
        print('模型保存成功, 路径为: %s' % save_path)


sess = tf.Session()
test = pd.read_csv('titanic/test.csv')
data_test = pre_pro_test(test)
predictions = sess.run(pred, feed_dict={x:data_test})
predictions = predictions.flatten()  # 二维数组变成一维数组
submission = pd.DataFrame({
    'PassengerId': data_test['PassengerId'], 'Survived': predictions
})
submission.to_csv('submission.csv', index=False)