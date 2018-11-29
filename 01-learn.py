import tensorflow as tf
m1 = tf.constant([[3,3]]) # shape:1*2
m2 = tf.constant([[2],[3]]) #shape: 2*1
product = tf.matmul(m1,m2)
#并不会出现结果
print(product)
#需要调用Session
sess = tf.Session()
#调用run方法
result = sess.run(product)
print(result)
sess.close()