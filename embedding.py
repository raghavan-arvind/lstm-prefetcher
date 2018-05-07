import numpy as np
import sys,os
import re
import pickle

DEBUG = True
INPUT_DELTA_CUTOFF = 50000 # try 50k

def debug(message):
    if DEBUG:
        sys.stdout.write(message)
        sys.stdout.flush()

# one hot encode a data set
def one_hot_encode(data):
    encode = dict()
    decode = dict()

    deltas = list(data)
    for i, d in enumerate(data):
        encode[d] = i
        decode[i] = d
    return encode, decode

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
    pattern = re.compile("\d+ \d+\s*\d*")
    prev = -1

    count = 0
    with open(filename, "r") as f:
        for line in f:
            line = re.sub('[\x00-\x1f]', '', line)
            if pattern.match(line.strip()):
                addr, pc = [int(s) for s in line.split()][:2]
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
    input_enc, input_dec = one_hot_encode(input_deltas)
    output_enc, output_dec = one_hot_encode(output_deltas)
    pcs_enc, pcs_dec = one_hot_encode(pcs)

    # clear up some memory
    del input_deltas
    del output_deltas
    del pcs

    # inputs and outputs to return
    trace_in_delta = []
    trace_in_pc = []
    trace_out_addr = []
    trace_out = []

    # build the current trace
    pattern = re.compile("\d+ \d+\s*\d*")
    prev = -1
    count = 1

    delta_list1 = []
    delta_list2 = []
    pc_list1 = []
    pc_list2 = []
    with open(filename, "r") as f:
        for line in f:
            line = re.sub('[\x00-\x1f]', '', line)
            if pattern.match(line):
                addr, pc = [int(s) for s in line.split()][:2]
                if prev != -1:
                    delta = addr - prev
                    
                    # add to encodings
                    delta_enc = enc(input_enc, delta)
                    pc_enc = enc(pcs_enc, pc)

                    ### list 1 ###
                    delta_list1.append(delta_enc)
                    pc_list1.append(pc_enc)
                    
                    # done with first list
                    if count % time_steps == 0:
                        trace_in_delta.extend(delta_list1)
                        trace_in_pc.extend(pc_list1)
                        delta_list1 = []
                        pc_list1 = []

                    # add output delta encoding
                    if count % time_steps == 1 and count > 64:
                        output_index = enc(output_enc, delta)
                        trace_out.append(output_index)
                        trace_out_addr.append(addr)

                    ### list 2 ###
                    # start list2 after 32 instructions
                    if count > time_steps/2:
                        delta_list2.append(delta_enc)
                        pc_list2.append(pc_enc)
                        
                        # done with second list
                        if count % time_steps == time_steps/2:
                            trace_in_delta.extend(delta_list2)
                            trace_in_pc.extend(pc_list2)
                            delta_list2 = []
                            pc_list2 = []

                        # add output delta encoding
                        if count % time_steps == time_steps/2+1 and count > time_steps/2 + 1:
                            output_index = enc(output_enc, delta)
                            trace_out.append(output_index)
                            trace_out_addr.append(addr)
                    
                    count += 1
                prev = addr
                if limit != -1 and count == limit:
                    break
    debug("done!\n")
    return trace_in_delta, trace_in_pc, trace_out_addr, trace_out, input_dec, output_dec


def split_training(trace_in_delta, trace_in_pc, trace_out, time_steps, train_ratio=0.70):
    cutoff_y = int(train_ratio*len(trace_out))
    cutoff_x = cutoff_y * time_steps

    train_x_delta = trace_in_delta[:cutoff_x]
    train_x_pc = trace_in_pc[:cutoff_x]
    train_y = trace_out[:cutoff_y]

    test_x_delta = trace_in_delta[cutoff_x:]
    test_x_pc = trace_in_pc[cutoff_x:]
    test_y = trace_out[cutoff_y:]

    return np.array(train_x_delta), np.array(train_x_pc), np.array(train_y), np.array(test_x_delta), np.array(test_x_pc), np.array(test_y)

def get_embeddings(filename, time_steps, train_ratio=0.70, lim=-1):
    deltas, pcs = crawl_deltas(filename, limit=lim)

    input_deltas = sorted([x for x in deltas.keys() if deltas[x] >= 10], key=lambda x: deltas[x], reverse=True)[:INPUT_DELTA_CUTOFF]

    size = min(50000, len(deltas.keys()))
    output_deltas = sorted(deltas.keys(), key=lambda x: deltas[x], reverse=True)[:size]

    trace_in_delta, trace_in_pc, trace_out_addr, trace_out, input_dec, output_dec = crawl_trace(filename, input_deltas, output_deltas, pcs, time_steps, limit=lim)

    debug("Created " + str(len(trace_out)) + " sets!\n")

    # ungodly return statement, but what can you do....
    return np.array(trace_in_delta), np.array(trace_in_pc), np.array(trace_out_addr), np.array(trace_out), len(input_deltas)+1, len(pcs)+1, len(output_deltas)+1, input_dec, output_dec

def dump_embedding(filename, time_steps, train_ratio=0.70, lim=-1):
    benchmark = filename[:filename.index(".")]
    with open(benchmark+"_embeddings.dump", "wb") as f:
        np.savez(f, get_embeddings(filename, time_steps, train_ratio=train_ratio, lim=lim))

# main testing
if __name__ == '__main__':
    trace_in_delta, trace_in_pc, trace_out_addr, trace_out, _, _, _, _, _ = get_embeddings(sys.argv[1], 64)
    split_training(trace_in_delta, trace_in_pc, trace_out, 64)
