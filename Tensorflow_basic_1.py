import tensorflow as tf

######
# 모델 구성 부분
######


X = tf.placeholder(tf.float32, [None, 3])
print(X)

# input data: placeholder X에 들어갈 입력 데이터
x_data = [[1,2,3], [4,5,6]]

# variables
W = tf.Variable(tf.random_normal([3,2]))
b = tf.Variable(tf.random_normal([2,1]))

expr = tf.matmul(X, W) + b


######
# 실행 부분
######
sess = tf.Session()
sess.run(tf.global_variables_initializer()) # variable initialize

print("== x_data ==")
print(x_data)
print("== W ==")
print(sess.run(W))
print("== b ==")
print(sess.run(b))
print("== expr ==")
print(sess.run(expr, feed_dict={X:x_data}))

sess.close()

