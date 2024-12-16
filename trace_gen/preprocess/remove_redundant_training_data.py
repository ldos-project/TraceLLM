import glob
import pickle
import sys
import time

from trace_gen.schema.call_graph_sequence import CallGraphDataSample, CallGraphDataSamples

def read_csv(filename, output_path, cg_hash_set):
    "converts a filename to a pandas dataframe"
    print(filename)
    cg_buffer: list[CallGraphDataSample] = []
    with open(filename, "r") as f:
        num_lines = len(f.readlines())
    with open(f"{output_path}/{filename.split('/')[-1]}", "w") as new_f:
        with open(filename, "r") as f:
            print("Start file reading.")
            start = time.time()
            cnt = 0
            for line_idx, cg_str in enumerate(f):
                cg_sample = CallGraphDataSample(raw_str=cg_str)
                if cg_sample.call_graph.valid:
                    cg_buffer.append(cg_sample)
                cnt += 1
                if cnt % 10000 == 0 or line_idx == num_lines - 1:
                    print(f"file reading progress: {cnt}, {time.time() - start:.2f}")
                    cg_samples = CallGraphDataSamples(cg_buffer)
                    for sample in cg_samples.samples:
                        if sample.call_graph.edge_dag.root.depth > 15:
                            continue

                        cg_hash = hash(sample.call_graph)
                        if cg_hash in cg_hash_set:
                            continue
                        cg_hash_set.add(cg_hash)

                        new_cg_str = f"00h00m00s000ms S_{str(sample.call_graph.service_id).zfill(9)} {'||'.join([e.to_edge_str_full_info() for e in sample.call_graph.edges])}"
                        new_f.write(new_cg_str + "\n")
                    cg_buffer = []

    return cg_hash_set

def main(dirname: str, output_path: str):
    sys.setrecursionlimit(10000)
    # merge
    csv_files = sorted(glob.glob(f"{dirname}/*.csv"))
    cg_hash_set = set()
    for file in csv_files: 
        cg_hash_set = read_csv(file, output_path, cg_hash_set=cg_hash_set)
        with open(f"{output_path}/cg_hash_set.pkl", "wb") as f:
            pickle.dump(cg_hash_set, f)

if __name__ == "__main__":
    dirname = "data/CallGraph/training_data"
    output_path = "data/CallGraph/training_data/deduplicated"
    main(dirname, output_path)
