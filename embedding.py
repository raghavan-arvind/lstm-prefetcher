import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import sys,os
import re

DEBUG = True
def debug(message):
    if DEBUG:
        sys.stdout.write(message)
        sys.stdout.flush()

# one hot encode a data set
def one_hot_encode(data):
    encode = dict()
    #decode = dict()

    deltas = list(data)
    for i, d in enumerate(data):
        encode[d] = i
        #decode[i] = d
    return encode #, decode

# used because unknown encodings should
# all go to the same index
def enc(encoder, element):
    if element in encoder:
        return encoder[element]
    else:
        return len(encoder)

# crawls the trace and returns a dictionary representing 
# the frequency of each delta
def crawl_deltas(filename, limit=-1):
    debug("Crawling " + filename + " for delta frequency... ")
    deltas = dict()
    pcs = set()
    pattern = re.compile("\d+ \d+")
    prev = -1

    count = 0
    with open(filename, "r") as f:
        for line in f:
            if pattern.match(line.strip()):
                addr, pc = [int(s) for s in line.strip().split()]
                if prev != -1:
                    delta = addr - prev
                    if delta in deltas:
                        deltas[delta] += 1
                    else:
                        deltas[delta] = 1
                    pcs.add(pc)
                    count += 1
                prev = addr
                if limit != -1 and count == limit:
                    break
    debug("done!\n")
    return deltas, pcs

# given list of input/output deltas, crawls the trace and creates
# a valid trace set
def crawl_trace(filename, input_deltas, output_deltas, pcs, time_steps, limit=-1):
    debug("Creating trace... ")
    input_deltas, output_deltas = set(input_deltas), set(output_deltas)

    # one-hot encodings for input/output deltas and pc
    input_enc = one_hot_encode(input_deltas)
    output_enc = one_hot_encode(output_deltas)
    pcs_enc = one_hot_encode(pcs)

    # clear up some memory
    del input_deltas
    del output_deltas
    del pcs

    # inputs and outputs to return
    trace_in = []
    trace_out = []

    # build the current trace
    cur_trace = []
    pattern = re.compile("\d+ \d+")
    prev = -1
    count = 0

    with open(filename, "r") as f:
        for line in f:
            if pattern.match(line.strip()):
                addr, pc = [int(s) for s in line.strip().split()]
                if prev != -1:
                    delta = addr - prev
                    cur_trace.append((delta, pc))
                    
                    # still using a sliding window because its the most
                    # efficient way to get sets of time_steps accesses
                    if len(cur_trace) == time_steps+1:
                        # no preprocessing anymore, invalid inputs map to
                        # the same encoding
                        '''
                        # check if all but the last are valid
                        input_valid = all(c[2] for c in cur_trace[:-1])
                        # check if last is valid
                        output_valid = cur_trace[time_steps][0] in output_enc
                        '''

                        for step in cur_trace[:-1]:
                            delta_index = enc(input_enc, step[0])
                            pc_index = enc(pcs_enc, step[1])
                            trace_in.append((delta_index, pc_index))
                        
                        output_step = cur_trace[time_steps]
                        output_index = enc(output_enc, output_step[0])

                        trace_out.append(output_index)

                        del cur_trace[0]
                    count += 1
                prev = addr
                if limit != -1 and count == limit:
                    break
    debug("done!\n")
    return trace_in, trace_out


def split_training(trace_in, trace_out, time_steps, train_ratio=0.70):
    cutoff_y = int(train_ratio*len(trace_out))
    cutoff_x = cutoff_y * time_steps

    train_x = trace_in[:cutoff_x]
    train_y = trace_out[:cutoff_y]

    test_x = trace_in[cutoff_x:]
    test_y = trace_out[cutoff_y:]

    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)

def get_embeddings(filename, time_steps, train_ratio=0.70, lim=-1):
    deltas, pcs = crawl_deltas(filename, limit=lim)

    input_deltas = sorted([x for x in deltas.keys() if deltas[x] >= 10], key=lambda x: deltas[x], reverse=True)
    size = min(50000, len(deltas.keys()))
    output_deltas = sorted(deltas.keys(), key=lambda x: deltas[x], reverse=True)[:size]

    trace_in, trace_out = crawl_trace(filename, input_deltas, output_deltas, pcs, time_steps, limit=lim)
    debug("Created " + str(len(trace_out)) + " sets!\n")

    # ungodly return statement, but what can you do....
    return np.array(trace_in), np.array(trace_out), len(input_deltas)+1, len(pcs)+1, len(output_deltas)+1


if __name__ == '__main__':
    trace_in, trace_out, num_inputs, num_pcs, num_output = get_embeddings(sys.argv[1], 20)
    print(trace_in)
    print("\n")
    print(trace_out)

    print(len(trace_in))
    print(len(trace_out))
