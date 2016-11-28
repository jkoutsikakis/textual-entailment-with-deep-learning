import numpy as np
import os
import cPickle

centroids_file_name = 'generated_centroids/generated_centroids.pickle'

class SNLI:

	def __init__(self):

		if(not os.path.isfile(centroids_file_name)):
			print("File not found: " + centroids_file_name)
			print("Please run generate_centroids.py first")
			sys.exit(1)

		with open(centroids_file_name, 'rb') as data_file:
			print("loading centroids...")
			self.train_dev_test_centroids = cPickle.load(data_file)

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

			if(new_pos >= len(self.train_dev_test_centroids[data_type]['sentences1'])):
				yield (self.train_dev_test_centroids[data_type]['sentences1'][pos:],
					self.train_dev_test_centroids[data_type]['sentences2'][pos:],
					self.train_dev_test_centroids[data_type]['labels'][pos:])
				break

			yield (self.train_dev_test_centroids[data_type]['sentences1'][pos:new_pos],
				self.train_dev_test_centroids[data_type]['sentences2'][pos:new_pos],
				self.train_dev_test_centroids[data_type]['labels'][pos:new_pos])

			pos = new_pos