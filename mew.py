import sae
import tensorflow as tf
import nus_wide as datamodel

training_epochs = 60
batch_size = 256
display_step = 1
examples_to_show = 20

mnist = datamodel.nus_wide_class(batch_size)

dims = [500, 128, 32]

graph = tf.Graph()
with graph.as_default():
	w_init_func = tf.random_normal
	b_init_func = tf.random_normal

	def loss_func(x,y):
		return 0.5 * tf.reduce_mean(tf.pow(x-y, 2 ) ) 

	reg_fac = 0.1
	act_func = tf.sigmoid

	sae = sae.SAE(dims, w_init_func, b_init_func, loss_func, act_func, input, reg_fac)

	input = tf.placeholder('float32', [None, dims[0]])
	loss = sae.loss_sae(input)

	v_error = sae.loss_sae(mnist.valid.images)

	optimizer = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(loss)

	init = tf.global_variables_initializer()
print('model constructed')

# Launch the graph
with tf.Session(graph=graph) as sess:
	last_loss = 0
	sess.run(init)
	total_batch = int(mnist.train.num_examples/batch_size)
	# Training cycle
	for epoch in range(training_epochs):
		# Loop over all batches
		for i in range(total_batch):
			batch_xs, batch_ys = mnist.train.next_batch()
			# Run optimization op (backprop) and cost op (to get loss value)
			_, c = sess.run([optimizer, loss], feed_dict={input: batch_xs})
		# Display logs per epoch step
		mnist.train.reset()
		if epoch % display_step == 0:
			[vl_error] = sess.run([v_error])
			print("Epoch:", '%04d' % (epoch+1),
				"cost=", "{:.9f}".format(c),
				'v_cost=', vl_error)
		# if(abs(last_loss - c) < 1e-5):
		# 	break
		last_loss = c

	print("Optimization Finished!")
