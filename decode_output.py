from embedding import get_embeddings, split_training
import sys,os,re,ast

DEBUG = True
def debug(message):
    if DEBUG:
        sys.stdout.write(message)
        sys.stdout.flush()



trace_dir = "/scratch/cluster/zshi17/ChampSimulator/CRCRealOutput/0426-LLC-trace/"
#trace_dir = ""

time_steps = 64

final_acc_str = "Final Testing Accuracy: "
input_delta_str = "Input Deltas:"
output_dec_str = "Output dec:"
excl_delta_str = "Make sure to exclude: "

# reads the output file
def read_output(filename):
    debug("Reading "+ trace_dir + filename+".out"+ " for predictions ...\n")
    
    accuracy = 0.0
    predictions = []
    input_deltas = set()
    excl_delta = 0

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
        elif reading_output_dec:
            output_dec_read = ast.literal_eval(line)
        elif reading_input_deltas:
            input_deltas = ast.literal_eval(line)

        reading_input_deltas = line.startswith(input_delta_str)
        reading_output_dec = line.startswith(output_dec_str)

    return accuracy, input_deltas, excl_delta, predictions, output_dec_read

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
    return num_correct / total


def eval_coverage(predictions, output_dec, excl_delta, correct_deltas, testing_addr):
    debug("Evaluating coverage ...\n")
    covered = [False] * len(testing_addr)

    repeats = [False] * len(testing_addr)
    for i in range(0, len(testing_addr)-window_size):
        if repeats[i]:
            continue
        cur_addr = testing_addr[i]
        window = testing_addr[i:i+window_size]
        for j, other_addr in enumerate(window):
            if cur_addr == other_addr:
                repeats[i+j] = True

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
    
    covered = [a and not b for a, b in zip(covered, repeats)]
    print(sum(repeats))
    print(len(repeats))
    print(sum(covered))
    return sum(covered) / (len(repeats)-sum(repeats))

if __name__ == '__main__':
    filename = trace_dir + sys.argv[1] + "_small.txt"
    trace_in_delta, trace_in_pc, trace_out_addr, trace_out, _, _, n_output_deltas, _, output_dec = get_embeddings(filename, time_steps)

    # trace out addr should have a 1-to-1 correlation without the number of output deltas
    err = "Deltas in output trace: %d\nAddresses in output trace: %d" % (len(trace_out), len(trace_out_addr))
    assert len(trace_out) == len(trace_out_addr), err

    # cut off the training set
    cutoff = int(len(trace_out) * 0.70)
    testing_addr = trace_out_addr[cutoff:]

    _, _, _, _, _, correct_deltas = split_training(trace_in_delta, trace_in_pc, trace_out, time_steps)

    # makes sure 1-to-1 correlation still exists
    err = "Deltas in ouput testing trace: %d\nAddresses in output testing trace: %d" % (len(correct_deltas), len(testing_addr))
    assert len(testing_addr) == len(correct_deltas), err

    accuracy, input_deltas, excl_delta, predictions, output_dec_read = read_output(sys.argv[1])
    assert output_dec == output_dec_read, "Output decoders don't match!"

    # make sure first delta lines up
    if correct_deltas[1] != excl_delta:
        err = "First 5 addresses %s\nFirst 5 deltas %s" % (trace_out_addr[:5], [output_dec[x] for x in trace_out[:5] if x != excl_delta])
        assert testing_addr[1] == testing_addr[0] + output_dec[correct_deltas[1]], err
    else:
        debug("Warning: can't check alignment of deltas w/ addresses!")


    recall = eval_recall(predictions, output_dec, excl_delta, input_deltas)
    coverage = eval_coverage(predictions, output_dec, excl_delta, correct_deltas, testing_addr)
    our_accuracy = eval_accuracy(predictions, output_dec, excl_delta, correct_deltas, testing_addr)

    print("precision: " + str(accuracy))
    print("recall: " + str(recall))
    print("degree: " + str(degree))
    print("coverage: " + str(coverage))
    print("accuracy: " + str(our_accuracy))
