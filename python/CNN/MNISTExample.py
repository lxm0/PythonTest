import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
x = tf.placeholder("float", [None, 784])#输入x
W = tf.Variable(tf.zeros([784,10]))#权重
b = tf.Variable(tf.zeros([10]))#偏置
y = tf.nn.softmax(tf.matmul(x,W) + b)#预测的标签

y_ = tf.placeholder("float", [None,10])#实际的标签

cross_entropy = -tf.reduce_sum(y_*tf.log(y))#计算

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)#梯度下降算法

init = tf.initialize_all_variables()#初始化所有变量

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    print (sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
