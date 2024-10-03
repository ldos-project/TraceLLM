from __future__ import annotations

from trace_gen.schema.call_graph import CallGraph
from trace_gen.schema.api_call import APICall
import random

class PromptRegistry:
    @classmethod
    def num_traces_prompt(cls, num_traces: int):
        return f"num traces:{num_traces}"
    
    @classmethod
    def must_include_prompt(cls, properties: list[str]):
        return f"include:({','.join(properties)})"
    
    @classmethod
    def sample_description_prompt(cls, call_graph: CallGraph):
        svc_id = f"S_{str(call_graph.service_id).zfill(9)}"
        num_edges = ""
        max_depth = ""
        if random.choices([True, False], weights=[9, 1])[0]:
            num_edges = f" with {len(call_graph.edges)} edges"
        if random.choices([True, False], weights=[9, 1])[0]:
            max_depth = f" with maximum depth {call_graph.edge_dag.root.depth}"
        return f"include:{svc_id}{num_edges}{max_depth}"

    @classmethod
    def resume_info_prompt(cls, edge: APICall):
        edge.rpc_id
        return f"begin id {edge.rpc_id} from {edge.src_ms} arrived {edge.timedelta}"
 
    @classmethod
    def start_time_prompt(cls, start: str):
        return f"start:{start}"
    
    @classmethod
    def end_time_prompt(cls, end: str):
        return f"end:{end}"

    @classmethod
    def arrived_time_prompt(cls, arrived: str):
        return f"arrived:{arrived}"
    
    @classmethod
    def num_edges_prompt(cls, call_graph: CallGraph):
        return f"num edges:{len(call_graph.edges)}"

    @classmethod
    def max_depth_prompt(cls, call_graph: CallGraph):
        return f"max depth:{call_graph.edge_dag.root.depth}"

    @classmethod
    def response_time_prompt(cls, call_graph: CallGraph, parent_edge: APICall):
        if parent_edge:
            return f"latency:{parent_edge.response_time}"
        return f"latency:{call_graph.edges[0].response_time}"

    @classmethod
    def service_id_prompt(cls, svc_id: str):
        return f"id:{svc_id}"

    @classmethod
    def include_svc_n_times_prompt(cls, svc_id: str, times: int):
        svc_name = f"S_{str(svc_id).zfill(9)}"
        return f"include id:{svc_name},times:{times}"
