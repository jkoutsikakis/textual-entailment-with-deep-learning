import numpy as np
import cPickle
import os
import sys
import math

def _calculateCentroid(sentence, idfs, embeddings):

	cent_nom = np.zeros(embeddings.shape[1])
	cent_den = 1

	seen = set()

	for index in sentence:
		if(index in seen):
			continue
		seen.add(index)
		tf = 0
		for i in sentence:
			if index == i:
				tf = tf + 1

		new = tf * idfs[index] if index in idfs else tf * idfs['unknown']

		cur_emp = embeddings[index]
		cent_nom = cent_nom + (new * cur_emp)
		cent_den = cent_den + new

	return cent_nom / cent_den

data_file_name = '../../data/preprocessed_data.cpkl'
embeddings_file_name = '../../data/preprocessed_embeddings.cpkl'

try: 
    os.makedirs('generated_centroids')
except OSError:
    if not os.path.isdir('generated_centroids'):
        raise

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
	special_embedding_indexes = cPickle.load(data_file)
	train_dev_test = cPickle.load(data_file)

with open(embeddings_file_name, 'rb') as embeddings_file:
	print("loading embeddings...")
	embeddings = cPickle.load(embeddings_file)

print("calculating idfs")

idfs = {}
for item in train_dev_test['train']:
	word_indexes = list(set(item['sentence1'])) + list(set(item['sentence2']))
	for index in word_indexes:
		idfs[index] = idfs[index] + 1 if index in idfs else 1
sent_card = len(train_dev_test['train']) * 2
for key in idfs.keys():
	idfs[key] = math.log(sent_card / idfs[key])
idfs['unknown'] = math.log(sent_card / 1)

print("generating centroids")

train_dev_test_centroids = {}

for data_type in train_dev_test:

	train_dev_test_centroids[data_type] = {}

	sentences1 = []
	sentences2 = []
	labels = []

	for item in train_dev_test[data_type]:
		
		sent1 = _calculateCentroid(item['sentence1'], idfs, embeddings)
		sent2 = _calculateCentroid(item['sentence2'], idfs, embeddings)

		sentences1.append(sent1)
		sentences2.append(sent2)
		labels.append(item['label'])

	train_dev_test_centroids[data_type]['sentences1'] = np.asarray(sentences1)
	train_dev_test_centroids[data_type]['sentences2'] = np.asarray(sentences2)
	train_dev_test_centroids[data_type]['labels'] = np.eye(3)[np.asarray(labels)]

print("dumping generated centroids...")
with open('generated_centroids/generated_centroids.pickle', 'wb') as save_file:
	cPickle.dump(train_dev_test_centroids, save_file)