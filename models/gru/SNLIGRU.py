import numpy as np
import cPickle
import os

embeddings_file_name = '../../data/preprocessed_embeddings.cpkl'
data_file_name = '../../data/preprocessed_data.cpkl'

class SNLI:

	def __init__(self):

		if(not os.path.isfile(data_file_name)):
			print("File not found: " + data_file_name)
			print("Please run preprocess_data.py first")
			sys.exit(1)

		if(not os.path.isfile(embeddings_file_name)):
			print("File not found: " + embeddings_file_name)
			print("Please run preprocess_data.py first")
			sys.exit(1)

		with open(data_file_name, 'rb') as data_file:
			print("loading data...")
			self.special_embedding_indexes = cPickle.load(data_file)
			self.train_dev_test = cPickle.load(data_file)

	def getEmbeddings(self):
		
		with open(embeddings_file_name, 'rb') as embeddings_file:
			print("loading embeddings...")
			embeddings = cPickle.load(embeddings_file)

		return embeddings

	def getTrainSize(self):
		return len(self.train_dev_test['train'])

	def getDevSize(self):
		return len(self.train_dev_test['dev'])

	def getTestSize(self):
		return len(self.train_dev_test['test'])

	def trainNextBatch(self, batch_size):
		for batch in self._next_batch(batch_size, 'train'):
			yield batch

	def devNextBatch(self, batch_size):
		for batch in self._next_batch(batch_size, 'dev'):
			yield batch

	def testNextBatch(self, batch_size):
		for batch in self._next_batch(batch_size, 'test'):
			yield batch

	def _next_batch(self, batch_size, data_type):
		pos = 0

		while True:

			new_pos = pos + batch_size

			if(new_pos >= len(self.train_dev_test[data_type])):
				yield (self._pad_sentences([d['sentence1'] for d in self.train_dev_test[data_type][pos:]]),
					self._pad_sentences([d['sentence2'] for d in self.train_dev_test[data_type][pos:]]),
					np.eye(3)[np.asarray([d['label'] for d in self.train_dev_test[data_type][pos:]])])
				break

			yield (self._pad_sentences([d['sentence1'] for d in self.train_dev_test[data_type][pos:new_pos]]),
				self._pad_sentences([d['sentence2'] for d in self.train_dev_test[data_type][pos:new_pos]]),
				np.eye(3)[np.asarray([d['label'] for d in self.train_dev_test[data_type][pos:new_pos]])])

			pos = new_pos

	def _pad_sentences(self, arr):
		max_len = 100

		new_arr = []
		len_arr = []
		for index in range(len(arr)):
			len_arr.append(len(arr[index]))
			new_arr.append(np.asarray(arr[index] + [self.special_embedding_indexes['ZERO_EMBEDDING_INDEX']] * (max_len - len(arr[index]))))

		return (np.asarray(new_arr), np.asarray(len_arr))