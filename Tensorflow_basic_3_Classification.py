import tensorflow as tf
import numpy as np

####
# 학습 데이터 정의
####


# [털, 날개] Yes : 1, No : 0
x_data = np.array([
        [0, 0], 
        [1, 0], 
        [1, 1], 
        [0, 0], 
        [0, 0], 
        [0, 1]
        ])
# [기타, 포유류, 조류] one-hot cording
y_data = np.array([
        [1, 0, 0], # 기타
        [0, 1, 0], # 포유류
        [0, 0, 1], # 조류
        [1, 0, 0], # 기타
        [1, 0, 0], # 기타
        [0, 0, 1]  # 조류
        ])


####
# 신경망 모델 구성
####

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random.uniform([2,3], -1.0, 1.0))
b = tf.Variable(tf.zeros([3]))

L = tf.add(tf.matmul(X,W), b) #L = (X*W + b)
L = tf.nn.relu(L) # L = Relu(X*W + b)

model = tf.nn.softmax(L)


###
# Cost(Loss) function
###

cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.math.log(model), axis = 1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)

####
# Training
####

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(100):
    sess.run(train_op, feed_dict={X:x_data, Y:y_data})

    # 학습 도중 10번에 한 번 loss 출력
    if (step +1) % 10 == 0:
        print(step+1, sess.run(cost, feed_dict={X:x_data, Y:y_data}))


####
# 결과 확인
####

prediction = tf.argmax(model, axis=1)
target = tf.argmax(Y, axis=1)
print('예측값:', sess.run(prediction, feed_dict={X:x_data}))
print('실측값:', sess.run(target, feed_dict={Y:y_data}))


# 정확도 출력
is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X:x_data, Y:y_data}))


