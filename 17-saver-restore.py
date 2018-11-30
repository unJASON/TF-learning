import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)
#批次大小
batch_size = 100
#计算共有多少个批次
n_batch = mnist.train.num_examples // batch_size
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

#网络架构
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
#线性判别函数,模式识别最基础的例子,最小化分类问题
prediction = tf.nn.softmax(tf.matmul(x,W)+b)

#loss
loss = tf.reduce_mean(tf.square(y-prediction))
#判断是否
correct_prediction = tf.equal(tf.arg_max(y,1),tf.arg_max(prediction,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#定义saver
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
    print("before acc:"+str(acc))
    saver.restore(sess,'net/my_net.ckpt')
    acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
    print("after acc:" + str(acc))