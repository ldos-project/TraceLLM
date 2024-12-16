import glob
import pickle
import time
from collections import defaultdict
from itertools import repeat
from multiprocessing import Pool

from trace_gen.schema.call_graph_sequence import (
    CallGraphDataSample,
    CallGraphDataSamples,
)


def read_csv(filename, output_path):
    "converts a filename to a pandas dataframe"
    print(filename)
    cg_buffer: list[CallGraphDataSample] = []
    cg_stats = defaultdict(int)
    svc_stats = defaultdict(int)
    latency_stats = defaultdict(list)
    communication_stats = defaultdict(lambda: defaultdict(int))
    infilling_stats = defaultdict(list)

    with open(filename, "r") as f:
        num_lines = len(f.readlines())

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

                    cg_stats[(len(sample.call_graph.edges), sample.call_graph.edge_dag.root.depth)] += 1
                    svc_stats[sample.call_graph.service_id] += 1
                    latency_stats[sample.call_graph.service_id].append(sample.call_graph.edges[0].finish_time)
                    for comm_tuple in set([(e.src_ms, e.rpc_type, e.dest_ms) for e in sample.call_graph.edges]):
                        src_ms, rpc_type, dest_ms = comm_tuple
                        infilling_stats[(sample.call_graph.service_id, src_ms, rpc_type)].append(dest_ms)
                    for comm_tuple in set([(e.src_ms, e.rpc_type, e.dest_ms) for e in sample.call_graph.edges]):
                        communication_stats[sample.call_graph.service_id][comm_tuple] += 1
                    communication_stats[sample.call_graph.service_id]["total"] += 1
                cg_buffer = []
        print(f"Write: {time.time() - start:.2f}")

        sorted_cg_stats = sorted(cg_stats.items(), key=lambda x: x[1], reverse=True)
        print("CG")
        print(sorted_cg_stats)
        print("SVC")
        sorted_svc_stats = sorted(svc_stats.items(), key=lambda x: x[1], reverse=True)
        filtered_svc_stats = [svc for svc in sorted_svc_stats if svc[1] >= 10]
        print(filtered_svc_stats)

        ### Calculate P90, rare communication events
        p90_stats = {}
        for svc, lat_list in latency_stats.items():
            if len(lat_list) >= 10:
                lat_list.sort()
                p90_index = int(len(lat_list) * 0.9)
                p90_value = lat_list[p90_index]
                p90_stats[svc] = (max(lat_list), p90_value, sum(lat_list) / len(lat_list))
        print("P90")
        print(p90_stats)
        with open(f"{output_path}/{filename.split('/')[-1]}.p90", "wb") as f:
            pickle.dump(p90_stats, f)

        rare_comm_events = defaultdict(list)
        for svc, comm_dict in communication_stats.items():
            for comm, count in comm_dict.items():
                if comm == "total":
                    continue
                if count / comm_dict["total"] < 0.1:
                    rare_comm_events[svc].append(comm)

        with open(f"{output_path}/{filename.split('/')[-1]}.rare_comm", "wb") as f:
            pickle.dump(rare_comm_events, f)

        with open(f"{output_path}/{filename.split('/')[-1]}.infilling_comm", "wb") as f:
            pickle.dump(infilling_stats, f)

def main(dirname: str, output_path: str):
    csv_files = sorted(glob.glob(f"{dirname}/*.csv"))

    with Pool(processes=50) as pool:  # or whatever your hardware can support
        pool.starmap(
            read_csv,
            zip(csv_files, repeat(output_path)),
        )

if __name__ == "__main__":
    dirname = "data/CallGraph/training_data/deduplicated"
    output_path = "data/CallGraph/training_data/cg_stats"
    main(dirname, output_path)
