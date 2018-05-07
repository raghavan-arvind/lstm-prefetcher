import sys, os
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from embedding import get_embeddings, split_training

def mean(x):
    return sum(x) * 1.0 / len(x)

# number of instructions per prediction
time_steps = 64

# number of nodes in the LSTM
num_units = 128

# learning rate
learning_rate = 0.001

# number of instructions per batch
batch_size = 1

# how many training/testing sets
train_ratio = 0.70

# embedding size
embedding_size = 128

# get datasets
trace_in_delta, trace_in_pc, trace_in_addr, trace_out, n_input_deltas, n_pcs, n_output_deltas, input_dec, output_dec  = get_embeddings(sys.argv[1], time_steps)
del trace_in_addr

# number of inputs to LSTM
n_input = n_input_deltas + n_pcs

# number of output deltas
n_classes = n_output_deltas 
assert n_classes <= 50001, "%d > 50001 output classes!" % n_classes


# split into training sets
# this is so ugly but what can you do
train_x_delta, train_x_pc, train_y, test_x_delta, test_x_pc, test_y = split_training(trace_in_delta, trace_in_pc, trace_out, time_steps, train_ratio=train_ratio)


# clear up space
del trace_in_delta
del trace_in_pc
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
x_delta = tf.placeholder("int32", [None, time_steps]) # None means t
x_pc = tf.placeholder("int32", [None, time_steps]) # None means t

# old one hot stuff
#x_delta_one_hot = tf.one_hot(x_delta, n_input_deltas, dtype=tf.int32)
#x_pc_one_hot = tf.one_hot(x_pc, n_pcs, dtype=tf.int32)
#x_one_hot_concat = tf.concat([x_delta_one_hot, x_pc_one_hot], 3)

embed_matrix_delta = tf.Variable(tf.random_normal([n_input_deltas, embedding_size]))
embed_matrix_pc = tf.Variable(tf.random_normal([n_pcs, embedding_size]))

embedded_delta = tf.nn.embedding_lookup(embed_matrix_delta, x_delta)
embedded_pc = tf.nn.embedding_lookup(embed_matrix_pc, x_pc)


embedded_concat = tf.concat((embedded_delta, embedded_pc), axis=2)

# input label placeholder
y = tf.placeholder("int32", [None, 1]) # takes it in as 1 dimensional
y_one_hot = tf.one_hot(y, n_classes)

# fix dimensions
y_final = tf.reshape(y_one_hot, (-1, n_classes))

# unstack inputs into sequence for static_rnn
inputs = tf.unstack(embedded_concat, time_steps, 1);

''' defining network '''
lstm_layer = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units)
outputs,_ = rnn.static_rnn(lstm_layer, inputs, dtype="float32")

prediction = tf.matmul(outputs[-1], out_weights) + out_bias

top_k = min(10, n_classes)

# loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y_final))
# optimization
opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# what if output delta is outside of top 50000?
# this catches that case and make sure it makes the prediction false
invalid_delta_encoding = tf.one_hot(tf.constant(n_classes-1), n_classes)
y_is_valid = tf.not_equal(tf.argmax(y_final, 1), tf.argmax(invalid_delta_encoding, 0))

# training accuracy -> trains on next memory access
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_final, 1))
correct_valid_prediction = tf.logical_and(y_is_valid, correct_prediction)
accuracy = tf.reduce_mean(tf.cast(correct_valid_prediction, tf.float32))

# testing accuracy -> tests top ten results
in_top_ten = tf.nn.in_top_k(prediction, tf.argmax(y_final, 1), top_k)
in_top_ten_valid = tf.logical_and(in_top_ten, y_is_valid)
accuracy_top_ten = tf.reduce_mean(tf.cast(in_top_ten_valid, tf.float32))

dim = test_y.reshape((-1, 1)).shape[0]
accuracy_testing = tf.Print(accuracy_top_ten, [tf.nn.top_k(prediction,k=top_k).indices], summarize=top_k*dim, message="Storing predictions for batch:\n")


print("\nOutput dec: ")
print(str(output_dec))

del output_dec

# initialize variables
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    iterator = 0

    while iterator<800 and (iterator+1)*batch_size < len(train_y):
        x_range = (iterator*batch_size*time_steps, (iterator+1)*batch_size*time_steps)
        batch_x_delta = train_x_delta[x_range[0]:x_range[1]]
        batch_x_delta = batch_x_delta.reshape((batch_size * time_steps))
        batch_x_delta = batch_x_delta.reshape((batch_size, time_steps))

        batch_x_pc = train_x_pc[x_range[0]:x_range[1]]
        batch_x_pc = batch_x_pc.reshape((batch_size * time_steps))
        batch_x_pc = batch_x_pc.reshape((batch_size, time_steps))

        batch_y = train_y[iterator*batch_size:(iterator+1)*batch_size]
        batch_y = batch_y.reshape((batch_size, 1))

        sess.run(opt, feed_dict={x_delta: batch_x_delta, x_pc: batch_x_pc, y: batch_y})

        if iterator % 10 == 0:
            acc=sess.run(accuracy,feed_dict={x_delta:batch_x_delta, x_pc:batch_x_pc,y:batch_y})
            los=sess.run(loss,feed_dict={x_delta:batch_x_delta, x_pc:batch_x_pc,y:batch_y})
            print("For iter ",iter)
            print("Accuracy ",acc)
            print("Loss ",los)
            print("__________________")
        iterator += 1


    # evaluate on training data set
    iterator = 0
    accuracies = []
    while (iterator+1)*batch_size < len(test_y):
        x_range = (iterator*batch_size*time_steps, (iterator+1)*batch_size*time_steps)
        batch_x_delta = test_x_delta[x_range[0]:x_range[1]]
        test_data_delta = batch_x_delta.reshape((batch_size * time_steps))
        test_data_delta = batch_x_delta.reshape((batch_size, time_steps))

        batch_x_pc = test_x_pc[x_range[0]:x_range[1]]
        test_data_pc = batch_x_pc.reshape((batch_size * time_steps))
        test_data_pc = batch_x_pc.reshape((batch_size, time_steps))

        batch_y = test_y[iterator*batch_size:(iterator+1)*batch_size]
        test_label = batch_y.reshape((batch_size, 1))
        accuracies.append(sess.run(accuracy_testing, feed_dict={x_delta: test_data_delta, x_pc: test_data_pc, y: test_label}))
        iterator += 1
    print("\nFinal Testing Accuracy: " + str(mean(accuracies)))

all_deltas_testing = set([input_dec[X] for X in test_x_delta if X != len(input_dec)])
print("\nInput Deltas: ")
print(all_deltas_testing)

print("\nMake sure to exclude: " + str(n_classes-1))
