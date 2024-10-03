from __future__ import annotations

from typing import List
from trace_gen.schema.call_graph import CallGraph
from trace_gen.schema.call_graph_sequence import CallGraphContext
from parse import parse

def instructions_to_dict(instructions: str, sep: str = "/") -> dict:
    instruction_dict = {}
    for instruction in instructions.split(sep):
        key, value = instruction.split(":")
        instruction_dict[key] = value

    return instruction_dict

class ValidationData:

    def __init__(self, context: str, call_graphs: List[CallGraph]):
        self.context = context
        self.call_graphs = call_graphs

    @property
    def context_kv(self):
        kv = {}
        for context in self.context.split("/"):
            key, value = context.split(":")
            if key == "num traces":
                kv[key] = int(value)
            elif key == "include":
                svc_desc = value.split(" with ")
                svc_id = svc_desc[0]
                num_edges = -1
                depth = -1
                for desc in svc_desc[1:]:
                    if "edges" in desc:
                        gt = parse("{num_edges} edges", desc)
                        num_edges = int(gt["num_edges"])
                    elif "maximum depth" in desc:
                        gt = parse("maximum depth {depth}", desc)
                        depth = int(gt["depth"])
                current_val = kv.get(key, {})
                current_val[int(svc_id.lstrip("S_"))] = (num_edges, depth)
                kv[key] = current_val
                # example - "include:(S_1,S_2)"
                # kv[key] = [int(s.lstrip("S_")) for s in value.lstrip("(").rstrip(")").split(",")]
            else:
                kv[key] = value
        return kv

    @classmethod
    def from_validation_sequence(cls, sequence: str) -> ValidationData:
        parsed = sequence.rstrip("\n").lstrip("<s>").rstrip("</s>").strip().split("[")
        return ValidationData(
            context=parsed[0],
            call_graphs=[
                CallGraph.get_call_graph_from_trace(trace.rstrip("]")) for trace in parsed[1:]
            ],
        )

    @classmethod
    def from_validation_sequences(cls, sequences: List[str]) -> List[ValidationData]:
        return [ValidationData.from_validation_sequence(seq) for seq in sequences]

class CGValidationData:

    def __init__(self, context: str, call_graph: CallGraph):
        self.context = context
        self.call_graph = call_graph

    @property
    def context_kv(self):
        kv = {}
        for context in self.context.split("/"):
            key, value = context.split(":")
            if key == "num edges" or key == "max depth" or "latency" in key:
                kv[key] = int(value)
            else:
                kv[key] = value
        return kv

    @classmethod
    def from_validation_sequence(cls, sequence: str) -> CGValidationData:
        parsed = sequence.rstrip("\n").lstrip("<s>").rstrip("</s>").strip().split("[")
        # something wrong here
        return CGValidationData(
            context=parsed[0],
            call_graph=CallGraph.get_call_graph_from_trace(parsed[1].rstrip("]")),
        )

    @classmethod
    def from_validation_sequences(cls, sequences: List[str]) -> List[CGValidationData]:
        return [CGValidationData.from_validation_sequence(seq) for seq in sequences]

class ContextValidationData:

    def __init__(self, query: str, context: CallGraphContext):
        self.query = query
        self.context = context

    @property
    def context_kv(self):
        kv = {}
        for context in self.context.split("/"):
            key, value = context.split(":")
            if key == "num edges" or key == "max depth" or "latency" in key:
                kv[key] = int(value)
            else:
                kv[key] = value
        return kv

    @classmethod
    def from_validation_sequence(cls, sequence: str) -> ContextValidationData:
        parsed = sequence.rstrip("\n").lstrip("<s> [GENERATE CONTEXTS]").rstrip("</s>").strip().split("{")
        # something wrong here
        return ContextValidationData(
            context=parsed[0],
            call_graph=CallGraphContext.from_context_string(parsed[1].rstrip("}").strip()),
        )

    @classmethod
    def from_validation_sequences(cls, sequences: List[str]) -> List[ContextValidationData]:
        return [ContextValidationData.from_validation_sequence(seq) for seq in sequences]
