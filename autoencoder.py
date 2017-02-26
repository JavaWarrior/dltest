# -*- coding: utf-8 -*-

""" Auto Encoder Example.
Using an auto encoder on MNIST handwritten digits.
References:
	Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
	learning applied to document recognition." Proceedings of the IEEE,
	86(11):2278-2324, November 1998.
Links:
	[MNIST Dataset] http://yann.lecun.com/exdb/mnist/
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datamodel

# Import MNIST data
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data", one_hot=True)


# Parameters
global_step = tf.Variable(0, trainable=False)
learning_rate = learning_rate = tf.train.exponential_decay(0.9, global_step,
                                           100000, 0.96, staircase=False)
training_epochs = 20
batch_size = 256
display_step = 1
examples_to_show = 10

mnist = datamodel.mnist_data_class(batch_size)

# Network Parameters
n_hidden_1 = 512 # 1st layer num features
n_hidden_2 = 265 # 2nd layer num features
n_hidden_3 = 128 # 2nd layer num features
n_input = 784 # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])

# weights = {
# 	'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
# 	'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
# 	'encoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
# 	'decoder_h1': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_2])),
# 	'decoder_h2': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
# 	'decoder_h3': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
# }
# biases = {
# 	'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
# 	'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
# 	'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
# 	'decoder_b1': tf.Variable(tf.random_normal([n_hidden_2])),
# 	'decoder_b2': tf.Variable(tf.random_normal([n_hidden_1])),
# 	'decoder_b3': tf.Variable(tf.random_normal([n_input])),
# }

weights = {
	'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
	'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_3])),
	'decoder_h1': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_1])),
	'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
	'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
	'encoder_b2': tf.Variable(tf.random_normal([n_hidden_3])),
	'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
	'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}

# weights = {
# 	'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
# 	'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_3])),
# 	'decoder_h1': tf.Variable(tf.zeros([n_hidden_3, n_hidden_1])),
# 	'decoder_h2': tf.Variable(tf.zeros([n_hidden_1, n_input])),
# }
# biases = {
# 	'encoder_b1': tf.Variable(tf.zeros([n_hidden_1])),
# 	'encoder_b2': tf.Variable(tf.zeros([n_hidden_3])),
# 	'decoder_b1': tf.Variable(tf.zeros([n_hidden_1])),
# 	'decoder_b2': tf.Variable(tf.zeros([n_input])),
# }


# # Building the encoder
# def encoder(x):
# 	# Encoder Hidden layer with sigmoid activation #1
# 	layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
# 								   biases['encoder_b1']))
# 	# Decoder Hidden layer with sigmoid activation #2
# 	layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
# 								   biases['encoder_b2']))
# 	layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
# 								   biases['encoder_b3']))
	
# 	return layer_3

def encoder(x):
	# Encoder Hidden layer with sigmoid activation #1
	layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['encoder_h1']),
								   biases['encoder_b1']))
	# Decoder Hidden layer with sigmoid activation #2
	layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
								   biases['encoder_b2']))
	
	return layer_2


# Building the decoder
def decoder(x):
	# Encoder Hidden layer with sigmoid activation #1
	layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['decoder_h1']),
								   biases['decoder_b1']))
	# Decoder Hidden layer with sigmoid activation #2
	layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
								   biases['decoder_b2']))
	return layer_2

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)


# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

valid = decoder(encoder(mnist.valid.images))
valid_error = 0.5 * tf.reduce_mean(tf.pow(tf.sub(valid, mnist.valid.images), 2))

# Define loss and optimizer, minimize the squared error
cost = tf.add(0.5 *  tf.reduce_mean(tf.pow(tf.sub(y_true, y_pred), 2  ) ), 
	0.3 * tf.add(
		tf.add(tf.reduce_mean(tf.pow(weights['encoder_h1'],2)),tf.reduce_mean( tf.pow(weights['encoder_h2'],2))),
		tf.add(tf.reduce_mean(tf.pow(weights['decoder_h1'],2)), tf.reduce_mean(tf.pow(weights['decoder_h2'],2)))
		)) 
# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, global_step = global_step)
# optimizer = tf.train.RMSPropOptimizer(0.001).minimize(cost)
# optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost, global_step = global_step)
# optimizer = tf.train.MomentumOptimizer(0.3, 0.9).minimize(cost)
# optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(cost)
print('model constructed ')


# Initializing the variables
init = tf.global_variables_initializer()

print('weights initialized')

# Launch the graph
with tf.Session() as sess:
	last_loss = 0
	sess.run(init)
	total_batch = int(mnist.train.num_examples/batch_size)
	# Training cycle
	for epoch in range(training_epochs):
		# Loop over all batches
		for i in range(total_batch):
			batch_xs, batch_ys = mnist.train.next_batch()
			# Run optimization op (backprop) and cost op (to get loss value)
			_, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
		# Display logs per epoch step
		mnist.train.reset()
		if epoch % display_step == 0:
			[v_error] = sess.run([valid_error])
			print("Epoch:", '%04d' % (epoch+1),
				"cost=", "{:.9f}".format(c),
				'v_cost=', v_error)
		# if(abs(last_loss - c) < 1e-5):
		# 	break
		last_loss = c

	print("Optimization Finished!")

	# Applying encode and decode over test set
	encode_decode, c = sess.run(
		[y_pred, cost], feed_dict={X: mnist.test.images[:examples_to_show]})
	# Compare original images with their reconstructions
	print('test error=', c)
	f, a = plt.subplots(2, 10, figsize=(10, 2))
	for i in range(examples_to_show):
		a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
		a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
	f.show()
	plt.draw()
	plt.waitforbuttonpress()
