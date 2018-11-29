import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt

#生成样本点

x_data = np.linspace(-0.5,0.5,200)[:,np.newaxis]#数据存放在第一维中,再给该数据升维,shape:[200,1]
noise = np.random.normal(0,0.2,x_data.shape)
y_data = np.square(x_data) + noise

#定义placeholder
x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])

#构建网络结构

#定义中间层
Weight_L1 = tf.Variable(tf.random_normal([1,10]))
biases_L1 = tf.Variable(tf.zeros([1,10]))
Wx_plus_b_L1 = tf.matmul(x,Weight_L1) + biases_L1
#activate function
L1 = tf.nn.tanh(Wx_plus_b_L1)

#定义输出层
Weight_L2 = tf.Variable(tf.random_normal([10,1]))
biases_L2 = tf.Variable(tf.zeros([1,1]))
Wx_plus_b_L2 = tf.matmul(L1,Weight_L2) + biases_L2
#预测
prediction = tf.nn.tanh(Wx_plus_b_L2)

loss = tf.reduce_mean(tf.square(y-prediction))
#训练
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(2000):
        sess.run(train_step , feed_dict={x:x_data,y:y_data})
    prediction_value = sess.run(prediction,feed_dict={x:x_data})
    #画图
    plt.figure()
    plt.scatter(x_data,y_data)
    plt.plot(x_data,prediction_value,'r-',lw = 5)
    plt.show()