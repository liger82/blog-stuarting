"""
In this file, we will implement back propagations by hands

We will use the Sigmoid Cross Entropy loss function.
This is equivalent to tf.nn.sigmoid_softmax_with_logits(logits, labels)

[Network Architecture]

Input: x
Layer1: x * W + b
Output layer = σ(Layer1)

Loss_i = - y * log(σ(Layer1)) - (1 - y) * log(1 - σ(Layer1))
Loss = tf.reduce_sum(Loss_i)

We want to compute that

dLoss/dW = ???
dLoss/db = ???

please read "Neural Net Backprop in one slide!" for deriving formulas

"""
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # for reproducibility

# Predicting animal type based on various features
xy = np.loadtxt('data/data-04-zoo.csv', delimiter=',', dtype=np.float32)
X_data = xy[:, 0:-1]
N = X_data.shape[0]
y_data = xy[:, [-1]]

# y_data has labels from 0 ~ 6
print("y has one of the following values")
print(np.unique(y_data))

# X_data.shape = (101, 16) => 101 samples, 16 features
# y_data.shape = (101, 1)  => 101 samples, 1 label
print("Shape of X data: ", X_data.shape)
print("Shape of y data: ", y_data.shape)

nb_classes = 7  # 0 ~ 6 까지로 변환한 동물 7종이 있음

# 16개의 feature를 지녔음
X = tf.placeholder(tf.float32, [None, 16])
y = tf.placeholder(tf.int32, [None, 1])  # 0 ~ 6

# 유력한 하나!
target = tf.one_hot(y, nb_classes)  # one hot
target = tf.reshape(target, [-1, nb_classes])
target = tf.cast(target, tf.float32)

W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# sigmoid function
def sigma(x):
    # σ(x) = 1 / (1 + exp(-x))
    return 1. / (1. + tf.exp(-x))

def sigma_prime(x):
    # derivative of the sigmoid function
    # 시그모이드 함수를 미분한 값 반환
    # σ'(x) = σ(x) * (1 - σ(x))
    return sigma(x) * (1. - sigma(x))

# Forward propagtion
layer_1 = tf.matmul(X, W) + b
y_pred = sigma(layer_1)

# Loss Function (end of forward propagation)
loss_i = - target * tf.log(y_pred) - (1. - target) * tf.log(1. - y_pred)
loss = tf.reduce_sum(loss_i)

# Dimension Check
# 이 조건이 충족되지 않으면 에러 발생
assert y_pred.shape.as_list() == target.shape.as_list()

## Back prop (chain rule)
# dE/da = (a-t)/(a(1-a))
d_loss = (y_pred - target) / (y_pred * (1. - y_pred) + 1e-7)
# sigmoid function을 미분한 값
d_sigma = sigma_prime(layer_1)
# dE/dl = dE/da * da/dl
d_layer = d_loss * d_sigma # == y_pred - target
d_b = d_layer # *1이 생략되어있음
# O = AW
# d_w = dE/dW임
# dE/dW = dE/dO * dO/dW 여기서 O=AW가 적용되어서
# dO/dW가 A.T가 됨 transpose되어서 자리좀 바꿈
# dE/dW = A.T. * dE/dO
# dE/dO = dE/dL*1 즉, d_layer와 같음
d_W = tf.matmul(tf.transpose(X), d_layer)

# Updating network using gradients
learning_rate = 0.01
# parameter를 update해줌
train_step = [
    tf.assign(W, W - learning_rate * d_W),
    tf.assign(b, b - learning_rate * tf.reduce_sum(d_b)),
]

# Prediction and Accuracy
prediction = tf.argmax(y_pred, 1)
# 목표값과 예측값이 같은지 아닌지
acct_mat = tf.equal(tf.argmax(y_pred, 1), tf.argmax(target, 1))
# 동일한지(0,1)를 평균낸 것이 정확도
acct_res = tf.reduce_mean(tf.cast(acct_mat, tf.float32))

# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(500):
        sess.run(train_step, feed_dict={X: X_data, y: y_data})

        if step % 10 == 0:
            # Within 300 steps, you should see an accuracy of 100%
            step_loss, acc = sess.run([loss, acct_res], feed_dict={
                                      X: X_data, y: y_data})
            print("Step: {:5}\t Loss: {:10.5f}\t Acc: {:.2%}" .format(
                step, step_loss, acc))

    # Let's see if we can predict
    pred = sess.run(prediction, feed_dict={X: X_data})
    for p, y in zip(pred, y_data):
        msg = "[{}]\t Prediction: {:d}\t True y: {:d}"
        print(msg.format(p == int(y[0]), p, int(y[0])))

'''
Output Example

Step:     0      Loss:  453.74799        Acc: 38.61%
Step:    20      Loss:   95.05664        Acc: 88.12%
Step:    40      Loss:   66.43570        Acc: 93.07%
Step:    60      Loss:   53.09288        Acc: 94.06%
...
Step:   290      Loss:   18.72972        Acc: 100.00%
Step:   300      Loss:   18.24953        Acc: 100.00%
Step:   310      Loss:   17.79592        Acc: 100.00%
...
[True]   Prediction: 0   True y: 0
[True]   Prediction: 0   True y: 0
[True]   Prediction: 3   True y: 3
[True]   Prediction: 0   True y: 0
...
'''
