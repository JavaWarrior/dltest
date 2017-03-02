import numpy as np
from six.moves import cPickle as pickle

image_size = 500
num_labels = 1000

train_img_path = 'trainImg.npy'
train_txt_path = 'trainTxt.npy'

test_img_path = 'testImg.npy'
test_txt_path = 'testTxt.npy'
test_gnd_path = 'testGnd.npy'

valid_img_path = 'validationImg.npy'
valid_txt_path = 'validationTxt.npy'
valid_gnd_path = 'validationGnd.npy'

data_path = 'data/'

def normalize(x):
	return (x - x.mean()) / x.var()


class nus_wide_class:
	def __init__(self, b_size):
		self.train = train_class(b_size,train_img_path, train_txt_path)
		self.test = test_class(test_img_path, test_txt_path, test_gnd_path)
		self.valid = valid_class(valid_img_path, valid_txt_path, valid_gnd_path)


class train_class:
	def __init__(self,b_size, img, txt):
		self.img = np.load(data_path + img).astype(np.float32)
		self.txt = np.load(data_path + txt).astype(np.float32)

		self.img = normalize(self.img)
		self.txt = normalize(self.txt)
		
		self.num_examples = len(self.txt)
		# print('training size', self.num_examples)
		self.current_batch = 0
		self.batch_size = b_size
	

	def next_batch(self):
		if(self.current_batch + self.batch_size < self.num_examples):
			placeholder = self.img[self.current_batch:self.current_batch + self.batch_size],self.txt[self.current_batch:self.current_batch + self.batch_size]
			self.current_batch = self.current_batch + self.batch_size
			return placeholder
		else:
			placeholder =  self.img[self.current_batch:],self.txt[self.current_batch:]
			self.current_batch = 0
			return placeholder
	def reset(self):
		self.current_batch = 0

class test_class:
	def __init__(self, img, txt, gnd):
		self.images = np.load(data_path + img).astype(np.float32)
		self.labels = np.load(data_path + txt).astype(np.float32)

		self.images = normalize(self.images)
		self.labels = normalize(self.labels)

		self.gnd = np.load(data_path + gnd).astype(np.float32)
		self.num_examples = len(self.labels)

class valid_class:
	def __init__(self, img, txt, gnd):
		self.images = np.load(data_path + img).astype(np.float32)
		self.labels = np.load(data_path + txt).astype(np.float32)

		self.images = normalize(self.images)
		self.labels = normalize(self.labels)

		self.gnd = np.load(data_path + gnd).astype(np.float32)
		self.num_examples = len(self.labels)


