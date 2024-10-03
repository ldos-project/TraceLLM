import glob
import pickle
import random
import time
from collections import defaultdict
from itertools import repeat
from multiprocessing import Pool

from trace_gen.schema.call_graph_sequence import (
    CallGraphDataSample,
    CallGraphDataSamples,
)
from trace_gen.generate.trace_oracle.schema.task_type import TraceGenTaskType
from itertools import chain


def read_csv(filename, output_path, cg_stat_path: str):
    "converts a filename to a pandas dataframe"
    print(filename)
    cg_buffer: list[CallGraphDataSample] = []
    cg_stats = defaultdict(int)
    depth_stats = defaultdict(int)
    svc_stats = defaultdict(int)
    latency_stats = defaultdict(list)
    communication_stats = defaultdict(lambda: defaultdict(int))

    task_type = TraceGenTaskType.graph_gen

    with open(f"{cg_stat_path}/merged_p90_stats.p90", "rb") as f:
        p90_stats = pickle.load(f)
    with open(f"{cg_stat_path}/merged_rare_comm_events.rare_comm", "rb") as f:
        rare_comm_events = pickle.load(f)

    with open(filename, "r") as f:
        print("Start file reading.")
        start = time.time()
        cnt = 0
        instruction_output = {"instruction": [], "output": []}
        for cg_str in f:
            cg_sample = CallGraphDataSample(raw_str=cg_str)
            if cg_sample.call_graph.valid:
                cg_buffer.append(cg_sample)
            cnt += 1
            if cnt % 10000 == 0:
                selected_cg_buffer = random.sample(cg_buffer, 500)
                print(f"file reading progress: {cnt}, {time.time() - start:.2f}")
                cg_samples = CallGraphDataSamples(selected_cg_buffer)
                for sample in cg_samples.samples:
                    if sample.call_graph.edge_dag.root.depth > 15:
                        continue

                    # Statistics
                    cg_stats[len(sample.call_graph.edges)] += 1
                    depth_stats[sample.call_graph.edge_dag.root.depth] += 1
                    svc_id = sample.call_graph.service_id
                    svc_stats[sample.call_graph.service_id] += 1
                    latency_stats[sample.call_graph.service_id].append(sample.call_graph.edges[0].finish_time)
                    for comm_tuple in set([(e.src_ms, e.rpc_type, e.dest_ms) for e in sample.call_graph.edges]):
                        communication_stats[sample.call_graph.service_id][comm_tuple] += 1
                    communication_stats[sample.call_graph.service_id]["total"] += 1

                    # CG Layers
                    cg_layers = sample.convert_to_call_graph_layers(generation_type=task_type)
                    cg_layer_list = list(chain(*[cg_layer.to_layer_list() for cg_layer in cg_layers]))
                    for idx, layer in enumerate(cg_layer_list):
                        tag = "split" if layer.merged else "layer"
                        layer_inst_output = layer.to_prompt(generation_type=task_type, p90_stats=p90_stats.get(svc_id, (0, 0, 0)), rare_comm_events=rare_comm_events.get(svc_id, []))
                        instruction_output["instruction"].append(layer_inst_output["instruction"])
                        instruction_output["output"].append(f"<{tag}>\n" + layer_inst_output["output"] + f"</{tag}>\n")
                cg_buffer = []
        open(f"{output_path}/{filename.split('/')[-1]}.instruction", "wb").write(pickle.dumps(instruction_output))
        print(f"Write: {time.time() - start:.2f}")

def main(dirname: str, output_path: str, cg_stat_path: str):
    # merge
    csv_files = sorted(glob.glob(f"{dirname}/*.csv"))
    with Pool(processes=50) as pool:  # or whatever your hardware can support
        pool.starmap(
            read_csv,
            zip(csv_files, repeat(output_path), repeat(cg_stat_path)),
        )


if __name__ == "__main__":
    dirname = "data/CallGraph"
    output_path = "data/CallGraph/output"
    cg_stat_path = "data/CallGraph/cg_stats/"
    main(dirname, output_path, cg_stat_path)
