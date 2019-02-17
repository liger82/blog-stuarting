# Lab 10 MNIST and Dropout
# 김성훈 교수님의 코드에 주석을 덧붙이고 조금 변경이 있습니다.
import tensorflow as tf
import random
# import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)  # reproducibility

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

# input place holders
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

# dropout (keep_prob) rate  0.7 on training, but should be 1 for testing
# keep_prob은 전체 weights 중 dropout하지 않고 남겨둘 비율을 의미한다.
# dropout은 training 때만 하므로 testing 때는 1로 한다.
# 이렇게 특정 타임별로 비율이 달라져서 placeholder로 작성한다.
keep_prob = tf.placeholder(tf.float32)

# ==== weights & bias for nn layers
# -- layer 1
# xavier_initializer
W1 = tf.get_variable("W1", shape=[784, 512],
                     initializer=tf.contrib.layers.xavier_initializer())
# 더 wide하게 구성					
b1 = tf.Variable(tf.random_normal([512]))
# activation function ; ReLU
# ReLU는 0보다 작으면 0, 0이상이면 항등함수 f(x) = x
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
# activation function을 거친 결과를 dropout의 input값으로 준다.
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

# -- layer 2
W2 = tf.get_variable("W2", shape=[512, 512],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([512]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

# -- layer 3
W3 = tf.get_variable("W3", shape=[512, 512],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([512]))
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

# -- layer 4
W4 = tf.get_variable("W4", shape=[512, 512],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([512]))
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

# -- layer 5
W5 = tf.get_variable("W5", shape=[512, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))
# 마지막에는 ReLU와 dropout을 적용하지 않는다.(유의할 것!!)
hypothesis = tf.matmul(L4, W5) + b5

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=hypothesis, labels=Y))
# adam이 성능이 굉장히 좋다!
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train my model
for epoch in range(training_epochs):
    avg_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.7}
        # c가 cost
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!\n')

# Test model and check accuracy
# argmax로 가장 큰 값을 지닌 인덱스, 즉 숫자를 뽑아와서 같은지 비교한다.
# 같으면 1, 다르면 0
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('정확도 Accuracy: ', sess.run(accuracy, feed_dict={
      X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1})) # keep_prob은 1로 준다.

# Get one and predict
# 하나 랜덤하게 뽑아서 테스트 해본다
r = random.randint(0, mnist.test.num_examples - 1)
print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
print("Prediction: ", sess.run(
    tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1], keep_prob: 1}))

# 보고 싶으면 풀어서 보세요~
# plt.imshow(mnist.test.images[r:r + 1].
#           reshape(28, 28), cmap='Greys', interpolation='nearest')
# plt.show()

'''
Epoch: 0001 cost = 0.451016781
Epoch: 0002 cost = 0.172922086
Epoch: 0003 cost = 0.127845718
Epoch: 0004 cost = 0.108389137
Epoch: 0005 cost = 0.093050506
Epoch: 0006 cost = 0.083469308
Epoch: 0007 cost = 0.075258198
Epoch: 0008 cost = 0.069615629
Epoch: 0009 cost = 0.063841542
Epoch: 0010 cost = 0.061475890
Epoch: 0011 cost = 0.058089914
Epoch: 0012 cost = 0.054294889
Epoch: 0013 cost = 0.048918156
Epoch: 0014 cost = 0.048411844
Epoch: 0015 cost = 0.046012261
Learning Finished!

정확도 Accuracy:  0.9796
Label:  [9]
Prediction:  [9]

'''
