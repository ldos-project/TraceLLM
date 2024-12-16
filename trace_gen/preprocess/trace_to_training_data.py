import glob
import os
from datetime import datetime, timedelta
from itertools import repeat
from multiprocessing import Pool
from typing import Dict, List

import pandas as pd

from trace_gen.schema.api_call import APICall
from trace_gen.schema.call_graph import CallGraph
import re


def join_valid_call_graph(edges: List[str]):
    api_calls = [APICall.get_api_call_from_trace(edge) for edge in edges]
    api_calls_with_unique_rpc_ids = []
    edges_with_unique_rpc_ids = []
    rpc_ids = set()
    for api_call, edge in zip(api_calls, edges):
        if api_call.rpc_id in rpc_ids:
            continue
        rpc_ids.add(api_call.rpc_id)
        api_calls_with_unique_rpc_ids.append(api_call)
        edges_with_unique_rpc_ids.append(edge)

    call_graph = CallGraph(
        arrived_at=datetime.min,
        service_id=0,
        edges=api_calls_with_unique_rpc_ids,
    )
    if call_graph.valid:
        edge_traces = "||".join(edges_with_unique_rpc_ids)
        pattern = re.compile("MS_(\d+)")
        ms_repl_fn = lambda m: f'MS_{m.group(1).zfill(5)}'
        return re.sub(pattern, ms_repl_fn, edge_traces)
    return None

def fill_svc_ids(svc: str):
    pattern = re.compile("S_(\d+)")
    svc_repl_fn = lambda m: f'S_{m.group(1).zfill(9)}'
    return re.sub(pattern, svc_repl_fn, svc)

def get_trace_ids_in_timestamp(df: pd.DataFrame):
    df_trace_ids = df.groupby("traceid").agg({"timestamp": "min"})
    return df_trace_ids.sort_values(by=["timestamp"])

def merge_trace_ids(df: pd.DataFrame, timestamp_by_traceid: Dict[str, int]):
    df_dict = df[["timestamp", "traceid"]].to_dict("index")
    for item in df_dict.values():
        tid = item["traceid"]
        base = timestamp_by_traceid[tid]
        item["delta"] = str(item["timestamp"] - base)
    df_delta = pd.DataFrame.from_dict(df_dict, orient="index")
    df = df.join(df_delta["delta"])
    df["sequence"] = (
        df["rpc_id"]
        + "|"
        + df["rpctype"]
        + "|"
        + df["um"]
        + "|"
        + df["dm"]
        + "|ARRIVED_"
        + df["delta"]
        + "|RT_"
        + df["rt"]
    )
    df = df.drop(["rpc_id", "rpctype", "dm", "um", "rt", "delta"], axis=1)
    # Need to check if passing "get_call_graph_hash" with "hash" key.
    df = df.groupby("traceid").agg(
        {"service": "min", "sequence": join_valid_call_graph}
    )

    cnt_valid_seq = df["sequence"].count()
    cnt_valid_svc = df["service"].count()
    print(f"Num valid call graphs: {cnt_valid_seq},{cnt_valid_svc},{cnt_valid_seq / cnt_valid_svc * 100}")
    return df.loc[df["sequence"].notnull()]


def read_csv(filename, base_date, date_format, output_path):
    "converts a filename to a pandas dataframe"
    print(filename)
    df = pd.read_csv(filename, on_bad_lines="skip")

    df = df[df["rpctype"].isin(["mc", "db", "http", "rpc", "mq"])]
    df = df[~df["um"].isin(["UNKNOWN", "UNAVAILABLE"])]
    df = df[~df["dm"].isin(["UNKNOWN", "UNAVAILABLE"])]

    df = df.loc[df["rt"].notnull()]
    df.loc[df["rt"].notnull(), "rt"] = df.loc[df["rt"].notnull(), "rt"].apply(
        lambda x: str(int(float(x)))
    )

    df = df.sort_values(by=["traceid", "timestamp", "rpc_id"])

    df = df.drop(["uminstanceid", "dminstanceid", "interface"], axis=1)
    df_trace_ids = get_trace_ids_in_timestamp(df)

    timestamp_by_traceid = df_trace_ids.to_dict()["timestamp"]
    df_merged = merge_trace_ids(df, timestamp_by_traceid)
    del df

    df_merged = df_merged.join(df_trace_ids["timestamp"])
    df_merged = df_merged.reindex(columns=["timestamp", "service", "sequence"])

    df_merged["timestamp"] = df_merged["timestamp"].apply(
        lambda x: (base_date + timedelta(milliseconds=x)).strftime(date_format)[:-3] + "ms"
    )
    df_merged = df_merged.sort_values(by=["timestamp"])
    df_merged["service"] = df_merged["service"].apply(lambda x: fill_svc_ids(x))

    path = output_path + filename.split("/")[-1]
    df_merged.to_csv(
        path,
        mode="a",
        header=False if os.path.isfile(path) else False,
        index=False,
        sep=" ",
    )


def main(dirname: str, output_path: str):
    base_date = datetime(year=2023, month=1, day=1)
    date_format = "%Hh%Mm%Ss%f"

    # merge
    csv_files = sorted(glob.glob(f"{dirname}/*.csv"))
    with Pool(processes=10) as pool:  # or whatever your hardware can support
        pool.starmap(
            read_csv,
            zip(csv_files, repeat(base_date), repeat(date_format), repeat(output_path)),
        )


if __name__ == "__main__":
    dirname = "data/CallGraph"
    output_path = "data/CallGraph/training_data/"
    main(dirname, output_path)
