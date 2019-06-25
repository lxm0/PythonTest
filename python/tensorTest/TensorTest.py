import tensorflow as tf
import numpy as np
#create data
x_data = np.random.rand(100).astype(np.float32)
print(x_data)
y_data = x_data*0.1 + 0.3
print(y_data)
#create tensorflow structure start
Weight = tf.Variable(tf.random_uniform([1],-1.0,1.0))
print(Weight)
biases = tf.Variable(tf.zeros([1]))
print(biases)
y = Weight*x_data +biases
print(y)
loss = tf.reduce_mean(tf.square(y-y_data))
optimizer  = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init  =tf.initialize_all_variables()

#create tensorflow structure end

sess = tf.Session()
sess.run(init)

# for step in range(201):
#     sess.run(train)
#     mnist.test.labels[:test_len]
#     if step%20 ==0:
#         print(step,sess.run(Weight),sess.run(biases))

