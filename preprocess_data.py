import json
import numpy as np
import nltk
import cPickle
import string
import os.path
import sys

embeddings_file = './raw/embeddings/glove.840B.300d.txt'

train_dev_test_files = {}
train_dev_test_files['train'] = './raw/dataset/snli_1.0_train.jsonl'
train_dev_test_files['dev'] = './raw/dataset/snli_1.0_dev.jsonl'
train_dev_test_files['test'] = './raw/dataset/snli_1.0_test.jsonl'

if(not os.path.isfile(embeddings_file)):
	print("File not found: " + embeddings_file)
	sys.exit(1)

if(not os.path.isfile(train_dev_test_files['train'])):
	print("File not found: " + train_dev_test_files['train'])
	sys.exit(1)

if(not os.path.isfile(train_dev_test_files['dev'])):
	print("File not found: " + train_dev_test_files['dev'])
	sys.exit(1)

if(not os.path.isfile(train_dev_test_files['test'])):
	print("File not found: " + train_dev_test_files['test'])
	sys.exit(1)

try: 
    os.makedirs('data')
except OSError:
    if not os.path.isdir('data'):
        raise

save_embeddings_file_name = './data/preprocessed_embeddings.cpkl'
save_data_file_name = './data/preprocessed_data.cpkl'

np.random.seed(seed=123)

embeddings_temp = {}

print("loading embeddings...")

with open(embeddings_file, 'r') as data_file:
	for line in data_file:
		cur = line.split(" ")
		embeddings_temp[cur[0]] = (np.asarray([float(i) for i in cur[1:]]))

print("preprocessing data...")

special_embedding_indexes = {}
special_embedding_indexes['FIRST_KNOWN_EMBEDDING_INDEX'] = 102
special_embedding_indexes['NULL_EMBEDDING_INDEX'] = 0
special_embedding_indexes['ZERO_EMBEDDING_INDEX'] = 1

current_embedding_index = special_embedding_indexes['FIRST_KNOWN_EMBEDDING_INDEX']

embedding_ids = {}
embeddings = []

embeddings.append(np.random.standard_normal(size=(300)) * 0.5)
embeddings.append(np.asarray([0] * 300))
for _ in range(0, 100):
	embeddings.append(np.random.standard_normal(size=(300)) * 0.5)

train_dev_test = { 'train' : [], 'dev' : [], 'test' : [] }
for file_name_id in train_dev_test_files:

	with open(train_dev_test_files[file_name_id], 'r') as data_file:

		data = []

		for line in data_file:
			single = json.loads(line)
			
			if single['gold_label'] == 'neutral':
				cur_label = 0
			elif single['gold_label'] == 'entailment':
				cur_label = 1
			elif single['gold_label'] == 'contradiction':
				cur_label = 2
			else:
				continue

			sent1 = nltk.word_tokenize(single['sentence1'])
			sent1[0] = str(sent1[0]).lower()
			sent1_emb = []
			for word in sent1:
				word = str(word).strip(string.punctuation)
				if word in embedding_ids:
					sent1_emb.append(embedding_ids[word])
				elif word in embeddings_temp:
					embeddings.append(embeddings_temp[word])
					embedding_ids[word] = current_embedding_index
					current_embedding_index = current_embedding_index + 1
					sent1_emb.append(embedding_ids[word])
				else:
					sent1_emb.append(2 + (hash(word) % 10 ** 2))

			sent2 = nltk.word_tokenize(single['sentence2'])
			sent2[0] = str(sent2[0]).lower()
			sent2_emb = []
			for word in sent2:
				word = str(word).strip(string.punctuation)
				if word in embedding_ids:
					sent2_emb.append(embedding_ids[word])
				elif word in embeddings_temp:
					embeddings.append(embeddings_temp[word])
					embedding_ids[word] = current_embedding_index
					current_embedding_index = current_embedding_index + 1
					sent2_emb.append(embedding_ids[word])
				else:
					sent2_emb.append(2 + (hash(word) % 10 ** 2))

			data.append({'sentence1': sent1_emb, 'sentence2': sent2_emb, 'label': cur_label})

	train_dev_test[file_name_id] = data

embeddings = np.asarray(embeddings)

print("dumping converted datasets...")
with open(save_embeddings_file_name, 'wb') as save_file:
	cPickle.dump(embeddings, save_file)

with open(save_data_file_name, 'wb') as save_file:
	cPickle.dump(special_embedding_indexes, save_file)
	cPickle.dump(train_dev_test, save_file)