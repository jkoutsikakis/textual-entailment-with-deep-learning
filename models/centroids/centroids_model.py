import tensorflow as tf
import numpy as np
from SNLICentroids import SNLI
import sys
import matplotlib.pyplot as plt
import os

def calculateAccuracy(batchIt, batch_size):
	count = 0
	acc = 0
	error = 0
	for batch_x_left, batch_x_right, batch_true_output in batchIt(batch_size):
		res = sess.run([cross_entropy, accuracy], feed_dict={x_left: batch_x_left, x_right: batch_x_right, true_output: batch_true_output, keep_prob: 1})
		card = len(batch_x_left)
		error = error + res[0] * card
		acc = acc + res[1] * card
		count = count + card
	return(error / float(count), acc / float(count))

try: 
    os.makedirs('saved_model')
except OSError:
    if not os.path.isdir('saved_model'):
        raise

try: 
    os.makedirs('results')
except OSError:
    if not os.path.isdir('results'):
        raise

try: 
    os.makedirs('summaries')
except OSError:
    if not os.path.isdir('summaries'):
        raise

snli = SNLI()

# Model Parameters
emb_size = 300
nlp_hidden_size = 200
n_classes = 3
learning_rate = 0.001
batch_size = 3000
display_epoch = 1
max_epochs = 300

# Define weights
weights = {
    'hidden1': tf.get_variable("weights_hidden1", shape=[emb_size * 2, nlp_hidden_size], initializer=tf.contrib.layers.xavier_initializer()),
    'hidden2': tf.get_variable("weights_hidden2", shape=[nlp_hidden_size, nlp_hidden_size], initializer=tf.contrib.layers.xavier_initializer()),
    'hidden3': tf.get_variable("weights_hidden3", shape=[nlp_hidden_size, nlp_hidden_size], initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable("weights_out", shape=[nlp_hidden_size, n_classes], initializer=tf.contrib.layers.xavier_initializer())
}

biases = {
    'hidden1': tf.Variable(tf.zeros_initializer([nlp_hidden_size])),
    'hidden2': tf.Variable(tf.zeros_initializer([nlp_hidden_size])), 
    'hidden3': tf.Variable(tf.zeros_initializer([nlp_hidden_size])),
    'out': tf.Variable(tf.zeros_initializer([n_classes]))
}

x_left = tf.placeholder(tf.float32, [None, emb_size])
x_right = tf.placeholder(tf.float32, [None, emb_size])

mlp_input = tf.concat(1, [x_left, x_right])
keep_prob = tf.placeholder(tf.float32)

h1 = tf.nn.relu(tf.matmul(mlp_input, weights['hidden1']) + biases['hidden1'])
h1_d = tf.nn.dropout(h1, keep_prob)
h2 = tf.nn.relu(tf.matmul(h1_d, weights['hidden2']) + biases['hidden2'])
h2_d = tf.nn.dropout(h2, keep_prob)
h3 = tf.nn.relu(tf.matmul(h2_d, weights['hidden3']) + biases['hidden3'])
h3_d = tf.nn.dropout(h3, keep_prob)

output = tf.matmul(h3_d, weights['out']) + biases['out']
true_output = tf.placeholder(tf.float32, [None, n_classes])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, true_output))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy) 

# Evaluate model
correct_pred = tf.equal(tf.argmax(output, 1), tf.argmax(true_output ,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

saver = tf.train.Saver()

results = {
	
	'train_loss': [],
	'dev_loss': [],
	'train_acc': [],
	'dev_acc': []

}

# Launch the graph

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5, allow_growth = True, allocator_type="BFC")
config_proto = tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=10)

with tf.Session(config=config_proto) as sess:

	summary_writer = tf.train.SummaryWriter('./summaries/', graph=sess.graph)
	summary_writer.flush()

	sess.run(init)

	cur_epoch = 1
	best_epoch = 0
	best_dev_accuracy = 0
	forgiven = 0
	max_forgiven = 20
	keep_prob_v = 0.8

	# Keep training until reach max iterations
	while cur_epoch <= max_epochs:

		print "Epoch: %d" % (cur_epoch)
		print "Training"

		for batch_x_left, batch_x_right, batch_true_output in snli.trainNextBatch(batch_size):
			res = sess.run([optimizer], feed_dict={x_left: batch_x_left, x_right: batch_x_right, true_output: batch_true_output, keep_prob: keep_prob_v})

		if cur_epoch % display_epoch == 0:

			print "Calculating Accuracy"
			
			train_res = calculateAccuracy(snli.trainNextBatch, batch_size)
			results['train_loss'].append(train_res[0])
			results['train_acc'].append(train_res[1])
			print "Train Loss= %f, Accuracy= %f" % (train_res[0], train_res[1])
			
			dev_res = calculateAccuracy(snli.devNextBatch, batch_size)
			results['dev_loss'].append(dev_res[0])
			results['dev_acc'].append(dev_res[1])
			print "Dev Loss= %f, Accuracy= %f" % (dev_res[0], dev_res[1])

			if(dev_res[1] > best_dev_accuracy):
				best_dev_accuracy = dev_res[1]
				best_epoch = cur_epoch
				forgiven = 0
				saver.save(sess, "./saved_model/saved_model.ckpt")
			else:
				forgiven = forgiven + 1

			if(max_forgiven == forgiven):
				break

		cur_epoch = cur_epoch + 1

	print "Optimization Finished! Best epoch %d" % (best_epoch)

# Launch the graph
with tf.Session(config=config_proto) as sess:
	saver.restore(sess, "./saved_model/saved_model.ckpt")
	test_res = calculateAccuracy(snli.testNextBatch, batch_size)
	print "Test Loss= %f, Accuracy= %f" % (test_res[0], test_res[1])

ep = np.asarray(list(range(display_epoch, len(results['train_loss']) * display_epoch + 1, display_epoch)))

fig = plt.figure()
plt.plot(ep, results['train_loss'], '-', linewidth=2, color='b', label='train loss')
plt.plot(ep, results['dev_loss'], '-', linewidth=2, color='g', label='dev loss')
#fig.suptitle('Loss', fontsize=20)
plt.xlabel('epochs', fontsize=18)
plt.ylabel('loss', fontsize=16)
plt.legend(bbox_to_anchor=(1, 1), loc='lower right', ncol=2)
#plt.show()
plt.savefig('./results/loss.png')

plt.clf()

fig = plt.figure()
plt.ylim(0, 1)
plt.plot(ep, results['train_acc'], '-', linewidth=2, color='b', label='train acc')
plt.plot(ep, results['dev_acc'], '-', linewidth=2, color='g', label='dev acc')
#fig.suptitle('Accuracy', fontsize=20)
plt.xlabel('epochs', fontsize=18)
plt.ylabel('accuracy', fontsize=16)
plt.legend(bbox_to_anchor=(1, 1), loc='lower right', ncol=2)
#plt.show()
plt.savefig('./results/acc.png')

with open('./results/results', 'w') as f:
	f.write('Epoch: ' + str(best_epoch) + '\n')
	f.write('Train Loss: ' + str(results['train_loss'][best_epoch - 1]) + ', Accuracy: ' + str(results['train_acc'][best_epoch - 1]) + '\n')
	f.write('Dev Loss: ' + str(results['dev_loss'][best_epoch - 1]) + ', Accuracy: ' + str(results['dev_acc'][best_epoch - 1]) + '\n')
	f.write('Test Loss: ' + str(test_res[0]) + ', Accuracy: ' + str(test_res[1]) + '\n')