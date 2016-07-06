from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


x_image = tf.reshape(x, [-1,28,28,1])

W_conv1a = weight_variable([3, 3, 1, 32])
b_conv1a = bias_variable([32])
h_conv1a = tf.nn.relu(conv2d(x_image, W_conv1a) + b_conv1a)
W_conv1b = weight_variable([3, 3, 32, 32])
b_conv1b = bias_variable([32])
h_conv1b = tf.nn.relu(conv2d(h_conv1a, W_conv1b) + b_conv1b)
h_pool1 = max_pool_2x2(h_conv1b)

W_conv2a = weight_variable([3, 3, 32, 64])
b_conv2a = bias_variable([64])
h_conv2a = tf.nn.relu(conv2d(h_pool1, W_conv2a) + b_conv2a)
W_conv2b = weight_variable([3, 3, 64, 64])
b_conv2b = bias_variable([64])
h_conv2b = tf.nn.relu(conv2d(h_conv2a, W_conv2b) + b_conv2b)
h_pool2 = max_pool_2x2(h_conv2b)

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def create_output_dir(out_dir):
    import os
    import datetime
    timestamp = "{:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
    if os.path.exists(out_dir):
      os.rename(out_dir, "%s_%s"%(out_dir, timestamp))
    os.mkdir(out_dir)

def get_test_accuracy():
    total_correct = 0.0
    batch_size = 100
    total = len(mnist.test.images)
    for j in range(total/batch_size):
        start = j*batch_size
        end = (j+1)*batch_size
        correct = correct_prediction.eval(feed_dict={
            x: mnist.test.images[start:end,:], 
            y_: mnist.test.labels[start:end,:], keep_prob: 1.0})
        total_correct += sum(correct)
    test_accuracy = total_correct/total
    return test_accuracy 


out_dir = "model_vgg_simple"
create_output_dir(out_dir)

saver = tf.train.Saver(max_to_keep=3)
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    max_test_accuracy = 0.0
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
            test_accuracy = get_test_accuracy()
            print("step %d, training accuracy %g, test accuracy %g"%(i, train_accuracy, test_accuracy))
            if test_accuracy > max_test_accuracy:
                max_test_accuracy = test_accuracy
                saver.save(sess, "%s/model_%g"%(out_dir,test_accuracy), global_step=i)

        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

