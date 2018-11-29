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
#loss = tf.reduce_mean(tf.square(y-prediction))
#loss 改成用交叉熵
loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction) )

#判断是否
correct_prediction = tf.equal(tf.arg_max(y,1),tf.arg_max(prediction,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})

        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Iter:"+str(epoch)+"acc:"+str(acc))
