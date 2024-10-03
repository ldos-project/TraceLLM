from __future__ import annotations

from typing import List, Optional
from itertools import chain
import random

from trace_gen.schema.api_call import APICall
from trace_gen.schema.call_graph_layer import CallGraphLayer

class APICallTrieNode:
    def __init__(self, char: str, parent: Optional[APICallTrieNode] = None):
        self.edges: List[APICall] = []
        self.char = char
        self.children = {}
        self.parent = parent
        self.microservice_trajectory = []

    def __hash__(self):
        if self.edges:
            edge = self.edges[0]
            children_hash = sum(
                [hash(child_node) for child_node in self.children.values()]
            )
            return (hash(edge) + children_hash) % (10**10)
        if len(self.children.keys()):
            children_hash = sum(
                [hash(child_node) for child_node in self.children.values()]
            )
            return (children_hash) % (10**10)
        return -1

    def set_edge(self, edge: APICall):
        self.edges.append(edge)

    def traverse(self):
        for child in self.children.values():
            yield from child.traverse()
        if self.edges:
            if len(self.microservice_trajectory) > 1:
                if self.microservice_trajectory[-2] != self.microservice_trajectory[-1]:
                    yield ((self.microservice_trajectory[-1], self.edges[0].src_ms), self.microservice_trajectory[-2], self.edges[0].dest_ms)
            elif len(self.microservice_trajectory) == 1:
                yield ((self.microservice_trajectory[-1], self.edges[0].src_ms), "None", self.edges[0].dest_ms)
            else:
                yield (("None", self.edges[0].src_ms), "None", self.edges[0].dest_ms)

    @property
    def depth(self) -> int:
        if self.children == {}:
            return 0
        return max(child.depth for child in self.children.values()) + 1

    def to_call_graph_layer_from_edges(self, service_id: str, latency: int, edges: list[APICall]) -> List[CallGraphLayer]:
        return [CallGraphLayer(
            service_id=service_id,
            latency=latency,
            edges=edges,
            child_layers=[]
        )]

    def to_call_graph_layer(self, service_id: str, latency: int, caller_node: str = "None") -> List[CallGraphLayer]:
        edges = list(chain(*[c.edges for c in self.children.values()]))
        if self.parent:
            child_layers = []
            for c in self.children.values():
                if c.children:
                    max_latency = -1
                    for edge in c.edges:
                        max_latency = max(max_latency, edge.timedelta + edge.response_time)
                    child_layers.append(c.to_call_graph_layer(service_id, max_latency, c.edges[0].src_ms))
        else:
            child_layers = [c.to_call_graph_layer(service_id, latency, c.edges[0].src_ms) for c in self.children.values() if c.children]
        child_layers = list(chain(*child_layers))

        edge_threshold = 15
        subgraph_threshold = 5
        def edge_chunk_to_cg_layer(edge_chunk):
            if len(edge_chunk) <= edge_threshold:
                rpc_ids_in_chunk = [e.rpc_id for e in edge_chunk]
                child_layers_in_chunk = [l for l in child_layers if l.start_rpc_id.rsplit(".", 1)[0] in rpc_ids_in_chunk]
                max_latency = max([e.timedelta + e.response_time for e in edge_chunk])
                return CallGraphLayer(
                    caller_node=caller_node,
                    service_id=service_id,
                    latency=max(latency, max_latency),
                    edges=edge_chunk,
                    child_layers=child_layers_in_chunk,
                )

        def merge_chunked_layers(cg_layers) -> list[CallGraphLayer]:
            merged_layers = []
            child_layers = []
            for cg_layer in cg_layers:
                child_layers.append(cg_layer)
                if len(child_layers) >= subgraph_threshold or cg_layer == cg_layers[-1]:
                    merged_layers.append(
                        CallGraphLayer(
                            caller_node=caller_node,
                            service_id=service_id,
                            latency=max([l.latency for l in child_layers]),
                            edges=list(chain(*[l.edges for l in child_layers])),
                            child_layers=child_layers,
                            merged=True,
                        )
                    )
                    child_layers = []
            if len(merged_layers) == 1:
                return merged_layers
            elif len(merged_layers) > subgraph_threshold:
                return merge_chunked_layers(merged_layers)
            return [
                CallGraphLayer(
                    caller_node=caller_node,
                    service_id=service_id,
                    latency=max([l.latency for l in merged_layers]),
                    edges=list(chain(*[l.edges for l in merged_layers])),
                    child_layers=merged_layers,
                    merged=True,
                )
            ]

        if len(edges) > edge_threshold:
            cg_layers = []
            edge_chunk = []
            chunk_size = random.randrange(int(edge_threshold* 4.0 / 5.0), edge_threshold)
            for edge in edges:
                edge_chunk.append(edge)
                if len(edge_chunk) > chunk_size or edge == edges[-1]:
                    cg_layers.append(edge_chunk_to_cg_layer(edge_chunk))
                    edge_chunk = []
            return merge_chunked_layers(cg_layers)

        edges = sorted(list(chain(*[c.edges for c in self.children.values()])))
        max_latency = max([e.timedelta + e.response_time for e in edges])
        cgl = CallGraphLayer(
                caller_node=caller_node,
                service_id=service_id,
                latency=max(max_latency, latency),
                edges=edges,
                child_layers=child_layers
            )
        for c in child_layers:
            c.set_parent(cgl)
        return [cgl]
    

class APICallTrie:
    def __init__(self, root: APICallTrieNode):
        self.root = root
        self.valid: bool = True
        self.invalid_reason: List[str] = []

    def __hash__(self):
        return hash(self.root)

    def insert(self, edge: APICall):
        trajectory = []
        node = self.root
        path = edge.rpc_id.split(".")
        for char in path:
            if char in node.children.keys():
                node = node.children[char]
                if node.edges:
                    trajectory.append(node.edges[0].src_ms)
            else:
                new_node = APICallTrieNode(char=char, parent=node)
                node.children[char] = new_node
                node = new_node
        node.edges.append(edge)
        node.microservice_trajectory = trajectory

    def validate(self, node: APICallTrieNode) -> None:
        if node == self.root:
            if "0" not in node.children.keys():
                self.invalid_reason += ["Root has no valid child"]
        else:
            """Put your validation logic here."""
            # 1. edges should not be empty
            if not node.edges and node.parent and node.parent.edges and node.children:
                self.invalid_reason += ["Edges are not connected"]
            else:
                # 2. get parents => compare response time
                parent_edges = node.parent.edges
                if parent_edges:
                    parent_rt = [edge.timedelta + edge.response_time for edge in parent_edges]
                    min_parent_rt = min(parent_rt)
                    child_rt = [edge.timedelta + edge.response_time for edge in node.edges]
                    max_child_rt = max(child_rt)
                    if max_child_rt > min_parent_rt:
                        self.invalid_reason += [
                            f"Parent's response time should be larger: {child_rt}, {parent_rt}"
                        ]

                    # 3. get parents => parent's dest == my src
                    relax_condition = set(("UNKNOWN", "UNAVAILABLE"))
                    parent_dest = set(
                        [edge.dest_ms for edge in parent_edges]
                    ).difference(relax_condition)
                    child_src = set([edge.src_ms for edge in node.edges]).difference(
                        relax_condition
                    )

                    if len(parent_dest) == 1 and len(child_src) == 1:
                        # TODO: handle "UNKNOWN" microservice ids
                        if parent_dest != child_src:
                            self.invalid_reason += [
                                f"Parent dest does not match Child src: {parent_dest}, {child_src}"
                            ]
                    elif len(parent_dest) <= 1 and len(child_src) <= 1:
                        pass
                    else:
                        self.invalid_reason += [
                            f"Parent/Child node has multiple dest/src: {parent_dest}, {child_src}"
                        ]

        for child in node.children.values():
            self.validate(child)

    @classmethod
    def from_edge_list(cls, edges: List[APICall]) -> APICallTrie:
        root = APICallTrieNode("")
        trie = APICallTrie(root)
        for edge in edges:
            trie.insert(edge)
        return trie

    def to_call_graph_layer(self, service_id: str, latency: int) -> List[CallGraphLayer]:
        return self.root.to_call_graph_layer(
            service_id=service_id,
            latency=latency,
        )
