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
x_image32 = tf.pad(x_image, [[0,0],[2,2],[2,2],[0,0]])

W_conv1 = weight_variable([3, 3, 1, 64])
b_conv1 = bias_variable([64])
h_conv1 = tf.nn.relu(conv2d(x_image32, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([3, 3, 64, 128])
b_conv2 = bias_variable([128])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = weight_variable([3, 3, 128, 256])
b_conv3 = bias_variable([256])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

W_conv4 = weight_variable([3, 3, 256, 512])
b_conv4 = bias_variable([512])
h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
h_pool4 = max_pool_2x2(h_conv4)

h_pool4_flat = tf.reshape(h_pool4, [-1, 2*2*512])
W_fc1 = weight_variable([2*2*512, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)
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


out_dir = "model_single3"
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

