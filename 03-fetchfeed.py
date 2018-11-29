import tensorflow as tf
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)

add = tf.add(input2,input3)
mul = tf.multiply(input1,add)
with tf.Session() as sess:
    #fetch 同时运行多个op
    result = sess.run([mul,add])
    print(result)

#feed
input1 = tf.placeholder(tf.float32) #32位浮点型占位符
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1,input2)
with tf.Session() as sess:
    #feed数据已字典形式传入
    print(sess.run(output,feed_dict={input1:[7.],input2:[2.]}))
