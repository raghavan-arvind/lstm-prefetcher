from embedding import get_embeddings, split_training
import sys,os,re,ast

DEBUG = True
def debug(message):
    if DEBUG:
        sys.stdout.write(message)
        sys.stdout.flush()

MAX_INS = 500000

def mean(x):
    return sum(x) / len(x)

#trace_dir = "/scratch/cluster/zshi17/ChampSimulator/CRCRealOutput/0426-LLC-trace/"
trace_dir = ""

time_steps = 64

final_acc_str = "Final Testing Accuracy: "
input_delta_str = "Input Deltas:"
output_dec_str = "Output dec:"
excl_delta_str = "Make sure to exclude: "
max_ins_str = "MAX_INS: "
batch_size_str = "Batch Size: "
retrains_str = "Retrains: "

# reads the output file
def read_output(filename):
    debug("Reading "+ trace_dir + filename+".out"+ " for predictions ...\n")
    
    accuracy = 0.0
    predictions = []
    input_deltas = set()
    excl_delta = 0
    max_ins = 0
    batch_size = 0
    retrains = 0

    reading_input_deltas = False
    reading_output_dec = False
    
    # crawl output
    output_file = open(trace_dir+filename+".out", "r")
    for line in output_file:
        # strip tensorflow nonsense
        if "CPU" in line and "2018" in line:
            line = line[:line.rfind("}")+1]

        # read predictions
        if line.startswith("[["):
            for step in line.split("]["):
                predictions.append([int(s) for s in re.sub("\\[|\\]", "", step).split()])
        elif line.startswith(final_acc_str):
            accuracy = float(line[len(final_acc_str):])
        elif line.startswith(excl_delta_str):
            excl_delta = int(line[len(excl_delta_str):])
        elif line.startswith(max_ins_str):
            max_ins = int(line[len(max_ins_str):])
        elif line.startswith(batch_size_str):
            batch_size = int(line[len(batch_size_str):])
        elif line.startswith(retrains_str):
            retrains = int(line[len(retrains_str):])
        elif reading_output_dec:
            output_dec_read = ast.literal_eval(line)
        elif reading_input_deltas:
            input_deltas = ast.literal_eval(line)

        reading_input_deltas = line.startswith(input_delta_str)
        reading_output_dec = line.startswith(output_dec_str)

    return accuracy, input_deltas, excl_delta, predictions, output_dec_read, max_ins, batch_size, retrains

def eval_recall(predictions, output_dec, excl_delta, output_deltas):
    debug("Evaluating recall ...\n")
    predicted_deltas = set()
    for top_k in predictions:
        for prediction in top_k:
            if prediction != excl_delta:
                predicted_deltas.add(output_dec[prediction])
    intersect = [x for x in predicted_deltas if x in set(output_deltas)]
    return len(intersect)/len(output_deltas)

degree = 1
window_size = 100

def eval_accuracy(predictions, output_dec, excl_delta, correct_deltas, testing_addr):
    debug("Evaluating accuracy ...\n")

    num_correct = 0
    total = 0
    for i in range(0, len(predictions)):
        if correct_deltas[i] != excl_delta:
            top_k = [output_dec[pred] for pred in predictions[i] if pred != excl_delta]
            if 0 in top_k:
                top_k.remove(0)
            top_k = top_k[0:degree]
            window = testing_addr[i:i+window_size]

            base_addr = testing_addr[i] - output_dec[correct_deltas[i]]
            predicted_addrs = set([base_addr+offset for offset in top_k])

            # sanity test
            if i > 0:
                err = "base address is not lined up, base addr = %s, base addr calc'd = %s" % (testing_addr[i-1], base_addr)
                assert testing_addr[i-1] == base_addr, err

            counts = [False] * degree
            for cur_addr in window:
                for ind, pred in enumerate(predicted_addrs):
                    if pred == cur_addr:
                        counts[ind] = True
                if sum(counts) == degree:
                    break
            total += degree
            num_correct += sum(counts)
    if total == 0:
        print(predictions)
        print(correct_deltas)
        print(testing_addr)

    return num_correct / total


def eval_coverage(predictions, output_dec, excl_delta, correct_deltas, testing_addr):
    debug("Evaluating coverage ...\n")
    covered = [False] * len(testing_addr)

    '''repeats = [False] * len(testing_addr)
    for i in range(0, len(testing_addr)-window_size):
        if repeats[i]:
            continue
        cur_addr = testing_addr[i]
        window = testing_addr[i:i+window_size]
        for j, other_addr in enumerate(window):
            if cur_addr == other_addr:
                repeats[i+j] = True'''

    # TODO: check if predictions are lined up
    for i in range(0, len(predictions)-window_size):
        if correct_deltas[i] != excl_delta:
            top_k = [output_dec[pred] for pred in predictions[i] if pred != excl_delta]
            if 0 in top_k:
                top_k.remove(0)
            top_k = top_k[0:degree]
            window = testing_addr[i:i+window_size]

            base_addr = testing_addr[i] - output_dec[correct_deltas[i]]
            predicted_addrs = set([base_addr+offset for offset in top_k])

            assert len(predicted_addrs) <= degree, "something wrong..."

            # sanity test
            if i > 0:
                err = "base address is not lined up, base addr = %s, base addr calc'd = %s\n%s\n%s" % (testing_addr[i-1], base_addr, testing_addr[i-1:i+2], correct_deltas[i-1:i+2])
                assert testing_addr[i-1] == base_addr, err

            for ind, cur_addr in enumerate(window):
                if cur_addr in predicted_addrs:
                    covered[i+ind] = True
    return sum(covered)/len(covered)
    '''covered = [a and not b for a, b in zip(covered, repeats)]
    print(sum(repeats))
    print(len(repeats))
    print(sum(covered))
    return sum(covered) / (len(repeats)-sum(repeats))'''

if __name__ == '__main__':
    filename = trace_dir + sys.argv[1] + ".txt"

    accuracy, input_deltas, excl_delta, predictions, output_dec_read, MAX_INS, batch_size, RETRAINS = read_output(sys.argv[1])
    assert MAX_INS != 0, "max ins not found in lstm output!"
    assert batch_size != 0, "batch size not found in lstm output!"
    assert RETRAINS != 0, "retrains not found in lstm output!"

    epoch = 0
    retrains = 0
    correct_deltas = []
    testing_addr = []
    trace_out = []
    trace_out_addr = []
    output_dec = None


    recalls = []
    coverages = []
    our_acc = []

    prediction_start = 0

    while True:
        trace_in_delta, trace_in_pc, trace_out_addr_epoch, trace_out_epoch, _, _, n_output_deltas, _, output_dec = get_embeddings(filename, time_steps, start=epoch*MAX_INS, lim=MAX_INS)

        if len(trace_out_epoch) == 0:
            retrains += 1
            epoch = 0
            prediction_start = 0

            if not retrains < RETRAINS:
                break

        trace_in_delta, trace_in_pc, trace_out_addr_epoch, trace_out_epoch, _, _, n_output_deltas, _, output_dec = get_embeddings(filename, time_steps, start=epoch*MAX_INS, lim=MAX_INS)

        # trace out addr should have a 1-to-1 correlation without the number of output deltas
        err = "Deltas in output trace: %d\nAddresses in output trace: %d" % (len(trace_out_epoch), len(trace_out_addr_epoch))
        assert len(trace_out_epoch) == len(trace_out_addr_epoch), err

        # cut off the training set
        cutoff = int(len(trace_out_epoch) * 0.70)
        testing_addr_epoch = trace_out_addr_epoch[cutoff:]

        batch_size_cutoff = len(testing_addr_epoch) % batch_size
        testing_addr_epoch = testing_addr_epoch[:-1 * batch_size_cutoff]

        _, _, _, _, _, correct_deltas_epoch = split_training(trace_in_delta, trace_in_pc, trace_out_epoch, time_steps, mod=batch_size)

        # makes sure 1-to-1 correlation still exists
        err = "Deltas in ouput testing trace: %d\nAddresses in output testing trace: %d" % (len(correct_deltas_epoch), len(testing_addr_epoch))
        assert len(testing_addr_epoch) == len(correct_deltas_epoch), err

        length = len(correct_deltas_epoch)
        coverage = eval_coverage(predictions[prediction_start:prediction_start+length], output_dec, excl_delta, correct_deltas_epoch, testing_addr_epoch)
        our_accuracy = eval_accuracy(predictions[prediction_start:prediction_start+length], output_dec, excl_delta, correct_deltas_epoch, testing_addr_epoch)

        if retrains == RETRAINS-1:
            coverages.append(coverage)
            our_acc.append(accuracy)

        prediction_start += length
        
        correct_deltas.extend(correct_deltas_epoch)
        testing_addr.extend(testing_addr_epoch)
        trace_out.extend(trace_out_epoch)
        trace_out_addr.extend(trace_out_addr_epoch)

        epoch += 1

    assert output_dec == output_dec_read, "Output decoders don't match!"

    #predictions = predictions[:len(correct_deltas)]
    #assert len(predictions) == len(correct_deltas), "# predictions = %d, # correct deltas = %d" % (len(predictions), len(correct_deltas))
    
    # make sure first delta lines up
    if correct_deltas[1] != excl_delta:
        err = "First 5 addresses %s\nFirst 5 deltas %s" % (trace_out_addr[:5], [output_dec[x] for x in trace_out[:5] if x != excl_delta])
        assert testing_addr[1] == testing_addr[0] + output_dec[correct_deltas[1]], err
    else:
        debug("Warning: can't check alignment of deltas w/ addresses!")


    recall = eval_recall(predictions, output_dec, excl_delta, input_deltas)
    #kcoverage = eval_coverage(predictions, output_dec, excl_delta, correct_deltas, testing_addr)
    #our_accuracy = eval_accuracy(predictions, output_dec, excl_delta, correct_deltas, testing_addr)

    print("precision: " + str(accuracy))
    print("recall: " + str(recall))
    print("degree: " + str(degree))
    print("coverage: " + str(mean(coverages)))
    print("accuracy: " + str(mean(our_acc)))
