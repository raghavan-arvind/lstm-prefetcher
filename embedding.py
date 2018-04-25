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

def one_hot_encode(deltas):
    encode = dict()
    decode = dict()

    deltas = list(deltas)
    for i, d in enumerate(deltas):
        encode[d] = i
        decode[i] = d
    return encode #, decode

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

    # clear up some virtual memory
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
                    cur_trace.append((delta, pc, delta in input_enc))
                    
                    if len(cur_trace) == time_steps+1:
                        # check if all but the last are valid
                        input_valid = all(c[2] for c in cur_trace[:-1])
                        # check if last is valid
                        output_valid = cur_trace[time_steps][0] in output_enc

                        if input_valid and output_valid:
                            for step in cur_trace[:-1]:
                                delta_enc = [False for i in range(len(input_enc))]
                                delta_enc[input_enc[step[0]]] = True

                                pc_enc = [False for i in range(len(pcs_enc))]
                                pc_enc[pcs_enc[step[1]]] = True

                                delta_enc.extend(pc_enc)
                                trace_in.append(delta_enc)
                            
                            output_step = cur_trace[time_steps]
                            output_step_enc = [False for i in range(len(output_enc))]
                            output_step_enc[output_enc[output_step[0]]] = True
                            trace_out.append(output_step_enc)

                        del cur_trace[0]
                    count += 1
                prev = addr
                if limit != -1 and count == limit:
                    break
    debug("done!\n")
    return trace_in, trace_out



def get_embeddings(filename, time_steps, train_ratio=0.70, lim=-1):
    deltas, pcs = crawl_deltas(filename, limit=lim)

    input_deltas = sorted([x for x in deltas.keys() if deltas[x] >= 10], key=lambda x: deltas[x], reverse=True)
    size = min(50000, len(deltas.keys()))
    output_deltas = sorted(deltas.keys(), key=lambda x: deltas[x], reverse=True)[:size]

    trace_in, trace_out = crawl_trace(filename, input_deltas, output_deltas, pcs, time_steps, limit=lim)
    debug("Created " + str(len(trace_out)) + " training sets!\n")

    cutoff_y = int(train_ratio*len(trace_out))
    cutoff_x = cutoff_y * time_steps

    train_x = trace_in[:cutoff_x]
    train_y = trace_out[:cutoff_y]

    test_x = trace_in[cutoff_x:]
    test_y = trace_out[cutoff_y:]

    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)


if __name__ == '__main__':
    get_embeddings(sys.argv[1], 20)

