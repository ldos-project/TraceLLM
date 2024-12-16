from __future__ import annotations

import random
from enum import Enum

from pydantic import BaseModel


class TraceGenTaskType(Enum):
    graph_gen = 1
    graph_gen_non_recursive = 2
    graph_gen_recursive = 3


class GraphGenInstruction(BaseModel):
    id: str = "S_0"
    start_rpc_id: str = "0"
    start_node: str = ""
    num_edges: int = -1
    depth: int = -1
    latency: int = -1
    offset: int = -1
    num_subgraphs: int = -1
    num_current_edges: int = -1

    @classmethod
    def from_prompt(cls, prompt: str) -> GraphGenInstruction:
        attributes = prompt.split("/")
        instruction = GraphGenInstruction()
        for attribute in attributes:
            if not attribute:
                continue
            key, value = attribute.split(":")
            if key == "id":
                instruction.id = value
            elif key == "start_rpc_id" or key == "start_edge_id":
                instruction.start_rpc_id = value
            elif key == "start_node":
                instruction.start_node = value
            elif key == "num_edges" or key == "num edges":
                instruction.num_edges = int(value)
            elif key == "max_depth" or key == "max depth" or key == "depth" or key == "remaining_depth":
                instruction.depth = int(value)
            elif key == "latency":
                instruction.latency = int(value)
            elif key == "offset" or key =="start_arrived" or key == "start_communication_at":
                instruction.offset = int(value)
            elif key == "num_subgraphs":
                instruction.num_subgraphs = int(value)
            elif key == "num_current_edges":
                instruction.num_current_edges = int(value)
        return instruction

    @property
    def valid(self):
        if (
            self.id
            and self.start_node
            and self.num_edges >= 0
            and self.depth >= 0
            and self.offset >= 0
        ):
            return True
        return False

    def to_prompt(self):
        attributes = [
            f"id:{self.id}",
            # f"start_rpc_id:{self.start_rpc_id}",
            f"start_node:{self.start_node}",
            f"num_edges:{self.num_edges}",
            f"remaining_depth:{self.depth}",
            f"latency:{self.latency}",
            f"offset:{self.offset}",
        ]
        random.shuffle(attributes)
        return "/".join(attributes)
