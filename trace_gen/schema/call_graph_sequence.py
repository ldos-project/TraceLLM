from __future__ import annotations

import random

from trace_gen.schema.call_graph import CallGraph
from trace_gen.schema.api_call import APICall
from trace_gen.train.dataset.prompt_registry import PromptRegistry
from itertools import chain
from typing import Optional
from trace_gen.generate.trace_oracle.schema.task_type import TraceGenTaskType

class CallGraphContext:
    def __init__(self, id: str, latency: int, depth: int, num_edges: int):
        self.id = id
        self.latency = latency
        self.depth = depth
        self.num_edges = num_edges

    @classmethod
    def from_context_string(cls, context_string: str) -> CallGraphContext:
        attributes = context_string.split("/")
        id = ""
        latency, depth, num_edges = -1, -1, -1
        for attribute in attributes:
            key, value = attribute.strip().split(":")
            if "id" in key:
                id = value
            elif "latency" in key:
                latency = int(value)
            elif "depth" in key:
                depth = int(value)
            elif "edge" in key:
                num_edges = int(value)

        return CallGraphContext(
            id=id,
            latency=latency,
            depth=depth,
            num_edges=num_edges,
        )

class CallGraphContextDesc:
    def __init__(self, query: str, contexts: list[CallGraphContext]):
        self.query = query
        self.contexts = contexts

    @classmethod
    def from_generated_window_file(self, window_fname: str) -> list[CallGraphContextDesc]:
        descriptions = []
        num_invalid = 0
        num_lines = 0
        with open(window_fname, "r") as f:
            for line in f:
                num_lines += 1
                query, context_str = line.split("{", 1)
                contexts = []
                for context in context_str.rstrip("\n").rstrip("</s>").split("}"):
                    if not context:
                        continue
                    try:
                        contexts.append(CallGraphContext.from_context_string(context.lstrip("{")))
                    except:
                        print(f"invalid: {context}")
                        num_invalid += 1
                        continue

                descriptions.append(
                    CallGraphContextDesc(
                    query = query.lstrip("<s> "),
                    contexts=contexts,
                ))
        print(f"num invalid: {num_invalid}, num lines: {num_lines} => {num_invalid/num_lines*100:.2f}")
        return descriptions

class CallGraphDataSample:
    def __init__(self, raw_str: str, parent_edge: Optional[APICall] = None):
        self.tokens = []
        self.raw_str = raw_str
        self.call_graph = CallGraph.get_call_graph_from_trace(self.raw_str)
        self.parent_edge = parent_edge

    @property
    def tokenized(self) -> bool:
        return bool(len(self.tokens))
    
    def convert_to_call_graph_layers(self, generation_type: TraceGenTaskType):
        """Break CallGraph into multiple CallGraphs."""
        edge_dag = self.call_graph.edge_dag
        # from root TrieNode => dag of layers
        if generation_type == TraceGenTaskType.graph_gen_non_recursive:
            return edge_dag.root.to_call_graph_layer_from_edges(
                service_id="S_" + f"{self.call_graph.service_id}".zfill(9),
                latency=self.call_graph.edges[0].response_time,
                edges=self.call_graph.edges,
            )
        else:
            return edge_dag.root.to_call_graph_layer(
                service_id="S_" + f"{self.call_graph.service_id}".zfill(9),
                latency=self.call_graph.edges[0].response_time,
            )
        

    def to_prompt_tokens(self, tokenizer, start_token: int, end_token: int, eos_token: int = 2) -> str:
        context_tokens = self.get_context_tokens(tokenizer=tokenizer)
        cg_tokens = [start_token] + self.tokens + [end_token]
        return context_tokens + cg_tokens + [eos_token]

    def get_context_kv(self):
        return {
            "num edges": len(self.call_graph.edges),
            "max depth": self.call_graph.edge_dag.root.depth,
            "latency": self.parent_edge.response_time if self.parent_edge else self.call_graph.edges[0].response_time,
        }

    def get_context(self, include_svc = False, include_arrived = False):
        contexts = []
        # if self.call_graph.edges[0].rpc_id != "0":
        #     contexts.append(PromptRegistry.resume_info_prompt(self.call_graph.edges[0]))
        if random.choices([True, False], weights=[9, 1])[0]:
            contexts.append(PromptRegistry.num_edges_prompt(call_graph=self.call_graph))
        if random.choices([True, False], weights=[9, 1])[0]:
            contexts.append(PromptRegistry.max_depth_prompt(call_graph=self.call_graph))
        if random.choices([True, False], weights=[9, 1])[0]:
            contexts.append(
                PromptRegistry.response_time_prompt(
                    call_graph=self.call_graph,
                    parent_edge=self.parent_edge
                )
            )
        if include_svc:
            contexts.append(PromptRegistry.service_id_prompt(svc_id=f"S_{str(self.call_graph.service_id).zfill(9)}"))
        if include_arrived:
            contexts.append(PromptRegistry.arrived_time_prompt(arrived=self.call_graph.arrived_at_prompt))
        random.shuffle(contexts)
        return "/".join(contexts)
    
    def get_context_tokens(self, tokenizer):
        # bos token is included
        return tokenizer.encode(self.get_context(include_svc=True))


class CallGraphContextWindow:
    def __init__(self, samples: list[CallGraphDataSample]):
        self.samples = samples

    def to_prompt_tokens(self, tokenizer, cg_tokens: list[int], eos_token: int = 2) -> str:
        context_tokens = self.get_context_tokens(tokenizer=tokenizer)
        return context_tokens + cg_tokens + [eos_token]

    def get_context(self):
        contexts = []
        contexts.append(PromptRegistry.start_time_prompt(start=self.samples[0].call_graph.arrived_at_prompt))
        if random.choices([True, False], weights=[9, 1])[0]:
            contexts.append(PromptRegistry.end_time_prompt(end=self.samples[-1].call_graph.arrived_at_prompt))
        if random.choices([True, False], weights=[9, 1])[0]:
            contexts.append(PromptRegistry.num_traces_prompt(num_traces=len(self.samples)))
        if random.choices([True, False], weights=[9, 1])[0]:
            unique_samples = []
            for sample in self.samples:
                if sample.call_graph.service_id in [s.call_graph.service_id for s in unique_samples]:
                    continue
                unique_samples.append(sample)
            selected_samples = random.sample(unique_samples, min(random.randint(1, len(unique_samples)), 5))
            contexts.append(PromptRegistry.must_include_prompt(["S_" + str(s.call_graph.service_id).zfill(9) for s in selected_samples]))

        random.shuffle(contexts)
        return "[GENERATE CONTEXTS]" + "/".join(contexts)
    
    def get_context_tokens(self, tokenizer):
        # bos token is included
        return tokenizer.encode(self.get_context())
    

class CallGraphSequence:
    def __init__(self, samples: list[CallGraphDataSample]):
        self.samples = samples

    def to_prompt_tokens(self, tokenizer, start_token: int, end_token: int, eos_token: int = 2) -> str:
        context_tokens = self.get_context_tokens(tokenizer=tokenizer)
        cg_tokens = list(
            chain(
                *[[start_token] + sample.tokens[:-1] + [end_token] for sample in self.samples]
            )
        )
        return context_tokens + cg_tokens + [eos_token]

    def get_context_kv(self):
        svc_desc = {}
        for s in self.samples:
            cg = s.call_graph
            attr = svc_desc.get(cg.service_id, [])
            attr.append((len(cg.edges), cg.edge_dag.root.depth))
            svc_desc[cg.service_id] = attr
        return {
            "num traces": len(self.samples),
            "include": svc_desc,
            "start": self.samples[0].call_graph.arrived_at_prompt,
            "end": self.samples[-1].call_graph.arrived_at_prompt,
        }

    def get_context(self):
        contexts = []
        # Include "start" always
        contexts.append(PromptRegistry.start_time_prompt(start=self.samples[0].call_graph.arrived_at_prompt))

        if random.choices([True, False], weights=[8, 2])[0]:
            contexts.append(PromptRegistry.end_time_prompt(end=self.samples[-1].call_graph.arrived_at_prompt))

        if random.choices([True, False], weights=[8, 2])[0]:
            contexts.append(PromptRegistry.num_traces_prompt(num_traces=len(self.samples)))

        if random.choices([True, False], weights=[9, 1])[0]:
            unique_samples = []
            for sample in self.samples:
                if sample.call_graph.service_id in [s.call_graph.service_id for s in unique_samples]:
                    continue
                unique_samples.append(sample)
            selected_samples = random.sample(unique_samples, min(random.randint(1, len(unique_samples)), 5))
            for sample in selected_samples:
                contexts.append(PromptRegistry.sample_description_prompt(call_graph=sample.call_graph))

        random.shuffle(contexts)
        return "/".join(contexts)
    
    def get_context_tokens(self, tokenizer):
        # bos token is included
        return tokenizer.encode(self.get_context())

class CallGraphDataSamples:

    def __init__(self, samples: list[CallGraphDataSample]):
        self.samples = samples

    def tokenize(self, tokenizer):
        samples_to_tokenize = [
            (idx, sample) for idx, sample in enumerate(self.samples) if not sample.tokenized
        ]
        tokenized_inputs = tokenizer(
            [sample.raw_str.split()[-1] for _, sample in samples_to_tokenize], truncation=False
        )["input_ids"]
        for idx, tokenized_input in enumerate(tokenized_inputs):
            sample_idx = samples_to_tokenize[idx][0]
            self.samples[sample_idx].tokens = tokenized_input[2:]

    def sequence_candidates(self, token_length: int) -> CallGraphSequence:
        """Choose random number of call graphs from the start of the list.
        
        The sum of token lengths should be equal to or smaller than token_length param.

        Note that the selected samples are going to be removed from the list.
        """
        match = 0
        max_idx = -1
        for idx, s in enumerate(self.samples):
            match += len(s.tokens)
            if match > token_length:
                max_idx = idx
                break

        if max_idx == 1 or max_idx == 0:  # One long call graph
            selected_samples = [self.samples[0]]
            self.samples = self.samples[1:]
        elif max_idx == -1: # The last step
            selected_samples = self.samples
            self.samples = []
        else:
            selected_idx = random.randint(1, max_idx-1)
            selected_samples = self.samples[:selected_idx]
            self.samples = self.samples[selected_idx:]
        return CallGraphSequence(samples=selected_samples)
