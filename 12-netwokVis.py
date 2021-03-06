import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)
#批次大小
batch_size = 100
#计算共有多少个批次
n_batch = mnist.train.num_examples // batch_size

#参数概要
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean',mean)#均值
        with tf.name_scope('stddev'):#标准差
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev',stddev)
        tf.summary.scalar('max',tf.reduce_max(var))#最大值
        tf.summary.scalar('min', tf.reduce_min(var))  # 最小值
        tf.summary.histogram('histogram',var)#直方图
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32,[None,784],name='x-input')
    y = tf.placeholder(tf.float32,[None,10],name='y-input')

#网络架构
with tf.name_scope('layer'):
    with tf.name_scope('weights'):
        W = tf.Variable(tf.zeros([784,10]),name='w')
        variable_summaries(W)
    with tf.name_scope('bias'):
        b = tf.Variable(tf.zeros([10]),name='b')
        variable_summaries(b)
#线性判别函数,模式识别最基础的例子,最小化分类问题
    with tf.name_scope('wx_plus_b'):
        wx_plus_b = tf.matmul(x, W) + b
    with tf.name_scope('softmax'):
        prediction = tf.nn.softmax(wx_plus_b)

#loss
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.square(y-prediction))
    tf.summary.scalar('loss',loss)
#判断是否
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.arg_max(y,1),tf.arg_max(prediction,1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        tf.summary.scalar('accuracy', accuracy)
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#合并所有summary
merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer=tf.summary.FileWriter('logs/',sess.graph)
    for epoch in range(50):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            summary,_=sess.run([merged,train_step],feed_dict={x:batch_xs,y:batch_ys})
            writer.add_summary(summary,epoch*n_batch+batch)
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Iter:"+str(epoch)+"acc:"+str(acc))
