import tensorflow as tf
import numpy as np
from SNLIAttention import SNLI
import sys
import matplotlib.pyplot as plt
import cPickle
import os

LOAD_PREVIOUS = (len(sys.argv) == 2 and sys.argv[1] == 'l')

def calculateAccuracy(batchIt, batch_size):
    count = 0
    acc = 0
    error = 0
    for batch_x_left, batch_x_right, batch_true_output in batchIt(batch_size):
        res = sess.run([cross_entropy, accuracy], feed_dict={x_left_p: batch_x_left[0], x_left_max_len: batch_x_left[1], x_right_p: batch_x_right[0], x_right_max_len: batch_x_right[1], true_output: batch_true_output, keep_prob: 1, smallest_number: np.asarray([-999999999])})
        card = len(batch_x_left[0])
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
emb_map_size = 200
nlp_hidden_size = 200
n_classes = 3
learning_rate = 0.001
batch_size = 1000
display_epoch = 1
max_epochs = 300

# Define weights
weights = {
    'map': tf.get_variable("weights_map", shape=[emb_size, emb_map_size], initializer=tf.contrib.layers.xavier_initializer()),
    'f1': tf.get_variable("f1", shape=[emb_map_size, nlp_hidden_size], initializer=tf.contrib.layers.xavier_initializer()),
    'f2': tf.get_variable("f2", shape=[nlp_hidden_size, nlp_hidden_size], initializer=tf.contrib.layers.xavier_initializer()),
    'g1': tf.get_variable("g1", shape=[emb_map_size * 2, nlp_hidden_size], initializer=tf.contrib.layers.xavier_initializer()),
    'g2': tf.get_variable("g2", shape=[nlp_hidden_size, nlp_hidden_size], initializer=tf.contrib.layers.xavier_initializer()),
    'h1': tf.get_variable("h1", shape=[emb_map_size * 2, nlp_hidden_size], initializer=tf.contrib.layers.xavier_initializer()),
    'h2': tf.get_variable("h2", shape=[nlp_hidden_size, nlp_hidden_size], initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable("weights_out", shape=[nlp_hidden_size, n_classes], initializer=tf.contrib.layers.xavier_initializer())
}

biases = {
    'map': tf.Variable(tf.zeros_initializer([emb_map_size])),
    'f1': tf.Variable(tf.zeros_initializer([nlp_hidden_size])),
    'f2': tf.Variable(tf.zeros_initializer([nlp_hidden_size])),
    'g1': tf.Variable(tf.zeros_initializer([nlp_hidden_size])),
    'g2': tf.Variable(tf.zeros_initializer([nlp_hidden_size])),
    'h1': tf.Variable(tf.zeros_initializer([nlp_hidden_size])),
    'h2': tf.Variable(tf.zeros_initializer([nlp_hidden_size])),
    'out': tf.Variable(tf.zeros_initializer([n_classes]))
}

global_step = tf.Variable(0)

embeddings = tf.Variable(snli.getEmbeddings(), trainable=False, dtype=tf.float32)

keep_prob = tf.placeholder(tf.float32)
smallest_number = tf.placeholder(tf.float32, [1])

#left input
x_left_p = tf.placeholder(tf.int32, [None, None])
x_left_max_len = tf.placeholder(tf.int32)
x_left = tf.nn.embedding_lookup(embeddings, x_left_p)

true_seq_size_mask_left = tf.reshape(tf.sign(tf.reduce_max(tf.abs(x_left), reduction_indices=2)), [-1, 1])
true_seq_size_inv_mask_left = (true_seq_size_mask_left * -1) + 1

x_left = tf.reshape(x_left, [-1, emb_size])
x_left = tf.matmul(x_left, weights['map']) + biases['map']
x_left = tf.mul(x_left, true_seq_size_mask_left)
x_left_batched = tf.reshape(x_left, [-1, x_left_max_len, emb_map_size])

#right input
x_right_p = tf.placeholder(tf.int32, [None, None])
x_right_max_len = tf.placeholder(tf.int32)
x_right = tf.nn.embedding_lookup(embeddings, x_right_p)

true_seq_size_mask_right = tf.reshape(tf.sign(tf.reduce_max(tf.abs(x_right), reduction_indices=2)), [-1, 1])
true_seq_size_inv_mask_right = (true_seq_size_mask_right * -1) + 1

x_right = tf.reshape(x_right, [-1, emb_size])
x_right = tf.matmul(x_right, weights['map']) + biases['map']
x_right = tf.mul(x_right, true_seq_size_mask_right)
x_right_batched = tf.reshape(x_right, [-1, x_right_max_len, emb_map_size])

#attention
att_left = tf.matmul(x_left, weights['f1']) + biases['f1']
att_left = tf.nn.relu(att_left)
att_left = tf.nn.dropout(att_left, keep_prob)
att_left = tf.matmul(att_left, weights['f2']) + biases['f2']
att_left = tf.nn.relu(att_left)
att_left = tf.nn.dropout(att_left, keep_prob)
att_left = tf.mul(att_left, true_seq_size_mask_left)
att_left = tf.reshape(att_left, [-1, x_left_max_len, emb_map_size])

att_right = tf.matmul(x_right, weights['f1']) + biases['f1']
att_right = tf.nn.relu(att_right)
att_right = tf.nn.dropout(att_right, keep_prob)
att_right = tf.matmul(att_right, weights['f2']) + biases['f2']
att_right = tf.nn.relu(att_right)
att_right = tf.nn.dropout(att_right, keep_prob)
att_right = tf.mul(att_right, true_seq_size_mask_right)
att_right = tf.reshape(att_right, [-1, x_right_max_len, emb_map_size])
 
attention = tf.batch_matmul(att_left, tf.transpose(att_right, [0, 2, 1]))

smallest_number_array = tf.reshape(tf.tile(smallest_number, [tf.size(attention)]), [-1, 1])

softmax_left = tf.reshape(attention, [-1, 1])
softmax_left = softmax_left + tf.mul(smallest_number_array, tf.reshape(tf.tile(tf.reshape(true_seq_size_inv_mask_right, [-1, x_right_max_len]), [1, x_left_max_len]), [-1, 1]))
softmax_left = tf.reshape(softmax_left, [-1, x_right_max_len])
softmax_left = tf.nn.softmax(softmax_left)
softmax_left = tf.reshape(softmax_left, [-1, 1])
softmax_left = tf.mul(softmax_left, tf.reshape(tf.tile(true_seq_size_mask_left, [1, x_right_max_len]), [-1, 1]))
softmax_left = tf.reshape(softmax_left, [-1, x_left_max_len, x_right_max_len])

softmax_right = tf.reshape(tf.transpose(attention, [0, 2, 1]), [-1, 1])
softmax_right = softmax_right + tf.mul(smallest_number_array, tf.reshape(tf.tile(tf.reshape(true_seq_size_inv_mask_left, [-1, x_left_max_len]), [1, x_right_max_len]), [-1, 1]))
softmax_right = tf.reshape(softmax_right, [-1, x_left_max_len])
softmax_right = tf.nn.softmax(softmax_right)
softmax_right = tf.reshape(softmax_right, [-1, 1])
softmax_right = tf.mul(softmax_right, tf.reshape(tf.tile(true_seq_size_mask_right, [1, x_left_max_len]), [-1, 1]))
softmax_right = tf.reshape(softmax_right, [-1, x_right_max_len, x_left_max_len])

phr_right = tf.batch_matmul(softmax_left, x_right_batched)
phr_left = tf.batch_matmul(softmax_right, x_left_batched)

g_input_left = tf.concat(2, [x_left_batched, phr_right])
g_input_left = tf.reshape(g_input_left, [-1, 2 * emb_map_size])

g_input_right = tf.concat(2, [x_right_batched, phr_left])
g_input_right = tf.reshape(g_input_right, [-1, 2 * emb_map_size])

v1 = tf.matmul(g_input_left, weights['g1']) + biases['g1']
v1 = tf.nn.relu(v1)
v1 = tf.nn.dropout(v1, keep_prob)
v1 = tf.matmul(v1, weights['g2']) + biases['g2']
v1 = tf.nn.relu(v1)
v1 = tf.nn.dropout(v1, keep_prob)
v1 = tf.mul(v1, true_seq_size_mask_left)
v1 = tf.reshape(v1, [-1, x_left_max_len, emb_map_size])
v1 = tf.reduce_sum(v1, 1)

v2 = tf.matmul(g_input_right, weights['g1']) + biases['g1']
v2 = tf.nn.relu(v2)
v2 = tf.nn.dropout(v2, keep_prob)
v2 = tf.matmul(v2, weights['g2']) + biases['g2']
v2 = tf.nn.relu(v2)
v2 = tf.nn.dropout(v2, keep_prob)
v2 = tf.mul(v2, true_seq_size_mask_right)
v2 = tf.reshape(v2, [-1, x_right_max_len, emb_map_size])
v2 = tf.reduce_sum(v2, 1)

h_input = tf.concat(1, [v1, v2])
h = tf.matmul(h_input, weights['h1']) + biases['h1']
h = tf.nn.relu(h)
h = tf.nn.dropout(h, keep_prob)
h = tf.matmul(h, weights['h2']) + biases['h2']
h = tf.nn.relu(h)
#h = tf.nn.dropout(h, keep_prob)
output = tf.matmul(h, weights['out']) + biases['out']

true_output = tf.placeholder(tf.float32, [None, n_classes])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, true_output))

m = tf.Print(cross_entropy, [global_step])

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy, aggregation_method=2, global_step=global_step, var_list=tf.trainable_variables())

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

    #summary_writer = tf.train.SummaryWriter('./summaries/', graph=sess.graph)
    #summary_writer.flush()

    sess.run(init)
    
    cur_epoch = 1
    best_epoch = 0
    best_dev_accuracy = 0
    forgiven = 0
    max_forgiven = 20
    keep_prob_v = 0.8

    if LOAD_PREVIOUS:
        saver.restore(sess, "./saved_model/saved_model.ckpt")
        with open("./saved_model/att_results.pickle", 'rb') as file:
            results = cPickle.load(file)
            best_epoch = cPickle.load(file)
            cur_epoch = best_epoch + 1
            best_dev_accuracy = cPickle.load(file)

    while cur_epoch <= max_epochs:

        print "Epoch: %d" % (cur_epoch)
        print "Training"

        for batch_x_left, batch_x_right, batch_true_output in snli.trainNextBatch(batch_size):
            res = sess.run([optimizer], feed_dict={x_left_p: batch_x_left[0], x_left_max_len: batch_x_left[1], x_right_p: batch_x_right[0], x_right_max_len: batch_x_right[1], true_output: batch_true_output, keep_prob: keep_prob_v, smallest_number: np.asarray([-999999999])})

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
                with open("./saved_model/att_results.pickle", 'wb') as file:
                    cPickle.dump(results, file)
                    cPickle.dump(best_epoch, file)
                    cPickle.dump(best_dev_accuracy, file)
            else:
                forgiven = forgiven + 1

            if(max_forgiven == forgiven):
                break

        cur_epoch = cur_epoch + 1

    print "Optimization Finished! Best acc %f Best epoch %d" % (best_dev_accuracy, best_epoch)

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
