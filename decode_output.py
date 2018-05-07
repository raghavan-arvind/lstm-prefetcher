from embedding import get_embeddings, split_training
import sys,os,re,ast

DEBUG = True
def debug(message):
    if DEBUG:
        sys.stdout.write(message)
        sys.stdout.flush()



trace_dir = "/scratch/cluster/zshi17/ChampSimulator/CRCRealOutput/0426-LLC-trace/"

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

degree = 2
window_size = 1000

def eval_accuracy(predictions, output_dec, excl_delta, correct_deltas, testing_addr):
    debug("Evaluating accuracy ...\n")
    num_correct = 0
    for i in range(0, len(predictions)):
        top_k = predictions[i][:degree]
        base_addr = testing_addr[i] - correct_deltas[i]

        # for sanity
        if i > 0:
            assert base_addr == testing_addr[i-1]
        predicted_addrs = [base_addr+output_dec[offset] for offset in top_k if offset != excl_delta]
        counts = [False] * degree
        for cur_addr in window:
            for ind, pred_addr in enumerate(predicted_addrs):
                if pred_addr == cur_addr:
                    counts[ind] = True
            if sum(counts) == degree:
                break
        num_correct += sum(counts)
    return num_correct / (degree * len(predictions))


def eval_coverage(predictions, output_dec, excl_delta, correct_deltas, testing_addr):
    debug("Evaluating coverage ...\n")
    covered = [False] * len(testing_addr)

    for i in range(0, len(predictions)-window_size):
        top_k = predictions[i][0:degree]
        window = testing_addr[i:i+window_size]

        base_addr = testing_addr[i] - correct_deltas[i]
        predicted_addrs = set([base_addr+output_dec[offset] for offset in top_k if offset != excl_delta])

        for ind, cur_addr in enumerate(window):
            if cur_addr in predicted_addrs:
                covered[i+ind] = True
            
    return sum(covered) / len(covered)

if __name__ == '__main__':
    filename = trace_dir + sys.argv[1] + ".txt"
    trace_in_delta, trace_in_pc, trace_in_addr, trace_out, _, _, n_output_deltas, _, output_dec = get_embeddings(filename, time_steps)
    assert len(trace_in_delta) == len(trace_in_addr)

    cutoff = int(len(trace_in_addr) * 0.70) * time_steps
    testing_addr = trace_in_addr[cutoff:]

    _, _, _, _, _, correct_deltas = split_training(trace_in_delta, trace_in_pc, trace_out, time_steps)
    print(len(testing_addr))
    print(len(correct_deltas))
    assert len(testing_addr) == len(correct_deltas)

    accuracy, input_deltas, excl_delta, predictions, output_dec_read = read_output(sys.argv[1])
    precision = eval_precision(predictions, output_dec, excl_delta, correct_deltas)
    recall = eval_recall(predictions, output_dec, excl_delta, input_deltas)
    assert(output_dec == output_dec_read)
    print("testing accuracy: " + str(accuracy))
    print("recall: " + str(recall))
    coverage = eval_coverage(predictions, output_dec, excl_delta, correct_deltas, testing_addr)
    print("coverage: " + str(coverage))
