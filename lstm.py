import sys, os
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from embedding import get_embeddings, split_training


# number of instructions per prediction
time_steps = 64

# number of nodes in the LSTM
num_units = 128

# learning rate
learning_rate = 0.001

# number of instructions per batch
batch_size = 128

# how many training/testing sets
train_ratio = 0.70

# get datasets
trace_in, trace_out, n_input_deltas, n_pcs, n_output_deltas  = get_embeddings(sys.argv[1], time_steps)

# number of inputs to LSTM
n_input = 2

# number of output deltas
n_classes = n_output_deltas 
assert n_classes <= 50001, "%d > 50001 output classes!" % n_classes


# split into training sets
train_x, train_y, test_x, test_y = split_training(trace_in, trace_out, time_steps, train_ratio=train_ratio)

# clear up space
del trace_in
del trace_out

# weights and biases of appropriate shape
out_weights = tf.Variable(tf.random_normal([num_units, n_classes]))
out_bias = tf.Variable(tf.random_normal([n_classes]))

''' 
defining placeholders 
None - means don't enforce batch_size, bc we use
       a different one for testing than for training 
'''
# input instruction placeholder
x = tf.placeholder("float", [None, time_steps, n_input]) # None means t
# input label placeholder
y = tf.placeholder("int64", [None, 1]) # takes it in as 1 dimensional
y_one_hot = tf.one_hot(y, n_classes)

# process input tensor to list [batch_size, n_inputs] of length time_steps
inputs = tf.unstack(x, time_steps, 1);

''' defining network '''
lstm_layer = rnn.BasicLSTMCell(num_units, forget_bias=1)
outputs,_ = rnn.static_rnn(lstm_layer, inputs, dtype="float32")

''' EDGE OF UNDERSTANDING '''
prediction = tf.matmul(outputs[-1], out_weights) + out_bias

# loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=prediction, labels=y_one_hot))
# optimization
opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# model evaluation
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# top 10 evaluation to match milad
#in_top_ten = tf.nn.in_top_k(prediction, batch_size, 1), y, 10)
#accuracy_top_ten = tf.reduce_mean(tf.cast(in_top_ten, tf.float32))

# initialize variables
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    iterator = 0
    #count = 0

    while iterator<800 and (iterator+1)*batch_size < len(train_y):
        x_range = (iterator*batch_size*time_steps, (iterator+1)*batch_size*time_steps)
        batch_x = train_x[x_range[0]:x_range[1]]
        batch_x = batch_x.reshape((batch_size * time_steps * n_input))
        batch_x = batch_x.reshape((batch_size, time_steps, n_input))

        batch_y = train_y[iterator*batch_size:(iterator+1)*batch_size]
        batch_y = batch_y.reshape((batch_size, 1))

        sess.run(opt, feed_dict={x: batch_x, y: batch_y})

        if iterator % 10 == 0:
            acc=sess.run(accuracy,feed_dict={x:batch_x,y:batch_y})
            los=sess.run(loss,feed_dict={x:batch_x,y:batch_y})
            print("For iter ",iter)
            print("Accuracy ",acc)
            print("Loss ",los)
            print("__________________")
        iterator += 1

        ''' used to loop data for testing purposes
        if (iterator+1)*batch_size >= len(train_y):
            iterator = 1
            count += 1

        if count == 100:
            break
        '''

    # evaluate on second half of data set
    test_data = test_x.reshape((-1, time_steps, n_input)) # -1 means make batch size the whole data set
    test_label = test_y.reshape((-1, 1))
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
