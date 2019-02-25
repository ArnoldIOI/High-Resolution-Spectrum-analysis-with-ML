import tensorflow as tf
from GenData import gendata

LR = 0.001


def loadData(filename):
    data_x = []
    data_y = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip().split(" ")
            data_x.append(line[:-1])
            data_y.append(line[-1])
    data_y = [[int(y)] for y in data_y]
    data_x = [[float(x_) for x_ in x] for x in data_x]
    # data = [x.append(y) for x, y in data_x, data_y]
    return data_x, data_y


train_x, train_y = loadData('./train_data.txt')
test_x, test_y = loadData('./test_data.txt')
te_x, te_y = loadData('./te_data.txt')

tf_x = tf.placeholder(tf.float32, [None, 100*100])
image = tf.reshape(tf_x, [-1, 100, 100, 1])
tf_y = tf.placeholder(tf.int32, [None, 1])

conv1 = tf.layers.conv2d(
    inputs=image,
    filters=16,
    kernel_size=5,
    strides=1,
    padding="same",
    activation=tf.nn.relu,
)       # -> (100, 100, 16)
pool1 = tf.layers.max_pooling2d(
    conv1,
    pool_size=5,
    strides=5,
)       # -> (20,20,16)
conv2 = tf.layers.conv2d(pool1, 32, 5, 1, 'same', activation=tf.nn.relu)  # -> (20,20,32)
pool2 = tf.layers.max_pooling2d(conv2, 5, 5)    # -> (4,4,32)
flat = tf.reshape(pool2, [-1, 4*4*32])
output = tf.layers.dense(flat, 1)

loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)
loss1 = tf.losses.absolute_difference(labels=tf_y, predictions=output)
train_op = tf.train.AdamOptimizer(LR).minimize(loss1)

accuracy = tf.metrics.accuracy(labels=tf_y, predictions=output)[1]

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)

for step in range(1000):
    tr_x = train_x[step*10:(step+1)*10]
    tr_y = train_y[step*10:(step+1)*10]
    _, loss_ = sess.run([train_op, loss1], {tf_x: tr_x, tf_y: tr_y})
    if step % 10 == 0:
        accuracy_, flat_representation = sess.run([accuracy, flat], {tf_x: test_x, tf_y: test_y})
        print('Step:', step, '| train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)
result = sess.run(output, feed_dict={tf_x: te_x})
print(result)


saver = tf.train.Saver()
model_path = "./model/model.ckpt"
save_path = saver.save(sess, model_path)


