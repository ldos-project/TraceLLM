from __future__ import annotations

from datetime import datetime
from typing import List, Optional

import math
from parse import parse
from pydantic import BaseModel, model_validator
from trace_gen.schema.api_call import APICall
from trace_gen.schema.api_call_trie import APICallTrie

class CallGraph(BaseModel):
    arrived_at: Optional[datetime] = None
    service_id: int
    edges: List[APICall]
    invalid_reason: List[str] = []

    @property
    def valid(self) -> bool:
        if len(self.invalid_reason) > 0:
            return False
        return True

    @property
    def edge_dag(self) -> APICallTrie:
        return APICallTrie.from_edge_list(self.edges)

    @property
    def arrived_at_prompt(self) -> str:
        """To HH$MM^SS&ms format."""
        return self.arrived_at.strftime("%Hh%Mm%Ss%f")[:-3] + "ms"

    def __hash__(self):
        # node => edges: (rpc_id, type, src, dest)
        return (self.service_id + hash(self.edge_dag)) % (10**10)

    @model_validator(mode="after")
    def check_call_graph(self) -> CallGraph:
        # check service id scope
        if self.service_id < 0:  # or self.service_id > MAX_SERVICE_ID:
            self.invalid_reason.append("invalid service id scope")

        for edge in self.edges:
            if edge.rpc_type in ["UNKNOWN", "UNAVAILABLE"] or edge.src_ms in ["UNKNOWN", "UNAVAILABLE"] or edge.dest_ms in ["UNKNOWN", "UNAVAILABLE"]:
                self.invalid_reason += ["missing info"]
                break
        edge_dag = self.edge_dag
        edge_dag.validate(edge_dag.root)
        self.invalid_reason += edge_dag.invalid_reason

    def split_into_chunks(self, num_min_chunks: int) -> list:
        # DFS
        root_parent = "0"
        sub_call_graphs = []
        edge_chunks = []
        start_edge = None
        threshold = math.ceil(len(self.edges) / num_min_chunks)
        for edge in self.edges:
            if root_parent not in edge.rpc_id or len(edge_chunks) >= threshold:
                sub_call_graphs.append((edge_chunks, start_edge))
                edge_chunks = []
                root_parent = edge.rpc_id.rsplit(".")[0]
                for start_edge_candidate in self.edges:
                    if start_edge_candidate.rpc_id == root_parent:
                        start_edge = start_edge_candidate
            edge_chunks.append(edge)
        if len(edge_chunks):
            sub_call_graphs.append((edge_chunks, start_edge))

        sub_cgs = []
        for edges, start_edge in sub_call_graphs:
            edges_str = "||".join(edge.to_edge_str() for edge in edges)
            sub_cgs.append((" ".join(["00h00m00s000ms", "S_" + f"{self.service_id}".zfill(9), edges_str]), start_edge))

        return sub_cgs

    @classmethod
    def get_call_graph_from_trace(cls, trace: str) -> CallGraph:
        columns = trace.split()
        if len(columns) != 3:
            return CallGraph(
                arrived_at=None, service_id=-1, edges=[], invalid_reason=["num columns"]
            )
        # datetime
        date_format_string = "{hr}h{min}m{sec}s{ms}ms"
        date_parsed = parse(date_format_string, columns[0])
        invalid_date_format = False
        try:
            arrived_at = datetime(
                year=2023,
                month=1,
                day=1,
                hour=int(date_parsed["hr"]),
                minute=int(date_parsed["min"]),
                second=int(date_parsed["sec"]),
                microsecond=int(date_parsed["ms"]) * 1000,
            )
        except ValueError:
            arrived_at = datetime(
                year=2023, month=1, day=1, hour=0, minute=0, second=0, microsecond=0
            )
            invalid_date_format = True
        except TypeError:
            arrived_at = datetime(
                year=2023, month=1, day=1, hour=0, minute=0, second=0, microsecond=0
            )
            invalid_date_format = True
        # service_id
        svc_id_fstring = "S_{svc_id}"
        svc_id_parsed = parse(svc_id_fstring, columns[1])
        svc_id = int(svc_id_parsed["svc_id"])

        # edges
        edges = []
        rpc_ids = set()
        for edge_string in columns[2].split("||"):
            api_call = APICall.get_api_call_from_trace(edge_string)
            if api_call.rpc_type in ["UNKNOWN", "UNAVAILABLE"] or api_call.src_ms in ["UNKNOWN", "UNAVAILABLE"] or api_call.dest_ms in ["UNKNOWN", "UNAVAILABLE"]:
                return CallGraph(
                    arrived_at=None, service_id=-1, edges=[], invalid_reason=["missing info"]
                )
            if api_call.rpc_id not in rpc_ids:
                edges.append(api_call)
                rpc_ids.add(api_call.rpc_id)

        # assume the last call graph generation is incomplete
        return CallGraph(
            arrived_at=arrived_at,
            service_id=svc_id,
            edges=sorted(edges),
            invalid_reason=["invalid datetime"] if invalid_date_format else [],
        )

    @classmethod
    def get_call_graphs_from_file(
        cls, fname: str, prefix_char: str = ""
    ) -> List[CallGraph]:
        call_graphs = []
        with open(fname, "r") as f:
            first_line = True
            while True:
                line = f.readline()
                if not line:
                    break
                if first_line and len(line) <= 1:
                    first_line = False
                    continue
                if prefix_char:
                    line = line.split(prefix_char, 1)[-1]
                call_graphs.append(CallGraph.get_call_graph_from_trace(line))
        return call_graphs

    @classmethod
    def get_call_graphs_from_inst_trace_file(
        cls, fname: str, start_symbol: str = "", limit: int = 1000,
    ) -> List[CallGraph]:
        call_graphs = []
        with open(fname, "r") as f:
            first_line = True
            while True:
                line = f.readline()
                if not line:
                    break
                if first_line and len(line) <= 1:
                    first_line = False
                    continue
                if start_symbol:
                    line = line.split(start_symbol, 1)[-1]
                instruction, cg_str = line.split("[", 1)
                # Find ID
                attributes = instruction.split("/")
                service_id = ""
                for attribute in attributes:
                    key, value = attribute.split(":")
                    if key == "id":
                        service_id = value
                # Parse call graph trace
                cg_str = cg_str.rsplit("]", 1)[0]
                cg_trace = f"00h00m00s000ms {service_id} {cg_str}"
                call_graphs.append(CallGraph.get_call_graph_from_trace(cg_trace))
                if len(call_graphs) >= limit:
                    break
        return call_graphs
