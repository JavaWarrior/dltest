import tensorflow as tf
import numpy as np

class AE:
	def __init__(self, h_dim, v_dim, w_init_func, b_init_func, act_func, reg_fac):
		self.h_dim = h_dim
		self.v_dim = v_dim

		self.w1 = tf.Variable(w_init_func([self.v_dim, self.h_dim]))
		self.w2= tf.Variable(w_init_func([self.h_dim, self.v_dim]))
		self.b1 = tf.Variable(b_init_func([self.h_dim]))
		self.b2 = tf.Variable(b_init_func([self.v_dim]))

		self.act_func = act_func

	def forward_step(self, input):
		return self.act_func( tf.add( tf.matmul(input, self.w1), self.b1 )) 

	def backward_step(self, input):
		return self.act_func( tf.add( tf.matmul( input, self.w2), self.b2 ) )

	def getreg(self):
		reg = tf.Variable(0.0)
		return tf.reduce_mean(tf.pow(self.w1, 2)) + tf.reduce_mean(tf.pow(self.w2, 2))
