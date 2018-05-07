import sys
from embedding import crawl_deltas, get_embeddings, split_training


for filename in sys.argv[1:]:
    trace_in_delta, trace_in_pc, trace_out, n_input_deltas, n_pcs, n_output_deltas, input_dec, output_dec  = get_embeddings(filename, 64)
    print("Input size for %s:\nNum input deltas: %d\nNum input pcs: %d" % (filename, n_input_deltas, n_pcs))
