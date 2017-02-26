from six.moves import cPickle as pickle
import numpy as np

pickle_file = 'notMNIST.pickle'
image_size = 28
num_labels = 10

def reformat(dataset, labels):
	dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
	# Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
	labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
	return dataset, labels

class mnist_data_class:
	def __init__(self, b_size):
		with open(pickle_file, 'rb') as f:
			save = pickle.load(f)
			# train_dataset = save['train_dataset']
			# train_labels = save['train_labels']
			# valid_dataset = save['valid_dataset']
			# valid_labels = save['valid_labels']
			# test_dataset = save['test_dataset']
			# test_labels = save['test_labels']
			self.train = train_class(b_size,(save['train_dataset'],save['train_labels']))
			self.test = test_class((save['test_dataset'],save['test_labels']))
			self.valid = valid_class((save['valid_dataset'],save['valid_labels']))
			del save  # hint to help gc free up memory
			# print('Training set', train_dataset.shape, train_labels.shape)
			# print('Validation set', valid_dataset.shape, valid_labels.shape)
			# print('Test set', test_dataset.shape, test_labels.shape)


class train_class:
	def __init__(self, b_size, data):
		self.dataset, self.labels = data
		self.dataset, self.labels = reformat(self.dataset, self.labels)
		self.num_examples = len(self.labels)
		# print('training size', self.num_examples)
		self.current_batch = 0
		self.batch_size = b_size

	def next_batch(self):
		if(self.current_batch + self.batch_size < self.num_examples):
			placeholder = self.dataset[self.current_batch:self.current_batch + self.batch_size],self.labels[self.current_batch:self.current_batch + self.batch_size]
			self.current_batch = self.current_batch + self.batch_size
			return placeholder
		else:
			placeholder =  self.dataset[self.current_batch:],self.labels[self.current_batch:]
			self.current_batch = 0
			return placeholder
	def reset(self):
		self.current_batch = 0

class test_class:
	def __init__(self, data):
		self.images, self.labels = data
		self.num_examples = len(self.labels)
		self.images, self.labels = reformat(self.images, self.labels)

class valid_class:
	def __init__(self, data):
		self.images, self.labels = data
		self.num_examples = len(self.labels)
		self.images, self.labels = reformat(self.images, self.labels)


