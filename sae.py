import ae
import tensorflow as tf

class SAE:
	def __init__(self, dims, w_init_func, b_init_func, loss_func, act_func, input, reg_fac):
		self.aes = []
		#dims [1000,500,32]
		for i in range(1,len(dims) ):
			self.aes.append(ae.AE(dims[i], dims[i-1], w_init_func, b_init_func, act_func, reg_fac))
		
		self.loss_func = loss_func

		self.reg_fac = reg_fac
	def loss_sae(self, input):
		reg = tf.Variable(0.0)
		for i in range(0, len(self.aes) ):
			if(i == 0):
				x = self.aes[i].forward_step(input)
			else:
				x = self.aes[i].forward_step(x)
			reg = reg +  self.aes[i].getreg()
		
		for i in range(len(self.aes) - 1, -1, -1):
			x = self.aes[i].backward_step(x)

		return self.loss_func(x	, input) + self.reg_fac * reg

	def loss_one_by_one(self, input):
		for i in range(0, len(self.aes)):
			if(i == 0):
				x = input
				y = self.aes[i].forward_step(x)
				y = self.aes[i].backward_step(y)
				l = loss_func(y,x)
			else:
				x = y
				y = self.aes[i].forward_step(x)
				y = self.aes[i].backward_step(y)
				l = tf.add(l, loss_func(y,x))
			l = l + self.aes[i].getreg() * self.reg_fac
	