from __future__ import annotations

from typing import List, Optional, Tuple

from pydantic import BaseModel, Field
import random
from itertools import chain

from trace_gen.schema.api_call import APICall
from trace_gen.generate.trace_oracle.schema.task_type import TraceGenTaskType
from trace_gen.utils.generation.prompt import (
    system_prompt,
    edges_to_prompt,
    num_current_edges_desc,
    start_rpc_id_in_edge_desc,
    num_generated_edges_desc,
    num_subgraphs_desc,
    num_remaining_edges_desc,
    subgraph_inst_latency_desc,
    subgraph_inst_remaining_depth_necessary_condition_desc,
    start_communication_at_in_edge_desc,
    subgraph_inst_offset_desc,
    subgraph_start_node_desc,
    subgraph_inst,
    generated_subgraph_inst_desc,
    split_subgraph_instructions_desc,
    splitted_subgraph_instruction_depth_desc,
    splitted_subgraph_instruction_start_node_desc,
    split_subgraph_instruction_latency_desc,
    split_subgraph_instruction_start_time_desc,
    subgraph_caller_desc,
    edge_finish_time_latency_desc,
    edge_start_finish_time_desc,
    edge_start_time_latency_desc,
    edge_start_time_requirement_latency_desc,
    edge_finish_time_requirement_latency_desc,
)

class CallGraphLayer(BaseModel):
    caller_node: str = "None"
    service_id: str
    latency: int
    edges: List[APICall]
    child_layers: List[CallGraphLayer]
    parent: Optional[CallGraphLayer] = Field(default=None)
    merged: bool = Field(default=False)

    def __hash__(self):
        edge_hash_sum = sum([edge.hash_wo_rpc_id() for edge in self.edges])
        child_structures = [(c.start_rpc_id,c.depth,c.num_edges) for c in self.child_layers]
        child_hash = sum([hash(c) % (10**10) for c in child_structures])
        return (hash(self.start_rpc_id) + edge_hash_sum + child_hash) % (10**10)

    @property
    def start_node(self) -> str:
        if self.edges:
            for edge in self.edges:
                if edge.src_ms != "UNKNOWN" and edge.src_ms != "UNAVAILABLE":
                    return edge.src_ms
        return "-1"

    @property
    def num_edges(self):
        self_num_edges = 0 if self.merged else len(self.edges)
        return self_num_edges + sum([l.num_edges for l in self.child_layers])
    
    @property
    def timedelta(self) -> int:
        if self.edges:
            return self.edges[0].timedelta
        return -1

    @property
    def start_rpc_id(self) -> str:
        if self.edges:
            return self.edges[0].rpc_id
        return "-1"

    @property
    def current_depth(self):
        max_depth = -1
        if self.edges:
            rpc_ids = [e.rpc_id for e in self.edges]
            for rpc_id in rpc_ids:
                depth = len(rpc_id.split('.'))
                max_depth = max(depth, max_depth)
        return max_depth

    @property
    def depth(self) -> str:
        if self.edges and self.child_layers:
            return max([l.depth for l in self.child_layers])
        return self.current_depth

    @property
    def recent_rpc_id(self):
        return '.'.join(self.start_rpc_id.rsplit('.', 2)[-2:])
    
    def to_layer_list(self):
        return [self] + list(chain(*[c.to_layer_list() for c in self.child_layers]))
    
    def set_parent(self, parent: CallGraphLayer):
        if not self.parent:
            self.parent = parent
            for c in self.child_layers:
                if self.merged:
                    c.set_parent(parent)
                else:
                    c.set_parent(self)
    
    def get_layer_attributes(self):
        attributes = []
        attributes.append(f"id:{self.service_id}")
        attributes.append(f"num_edges:{self.num_edges}")
        # if self.depth:
        #     attributes.append(f"max_depth:{self.depth}")
        attributes.append(f"start_node:{self.start_node}")
        attributes.append(f"start_edge_id:{self.recent_rpc_id.split('.')[-1]}")
        attributes.append(f"start_communication_at:{self.timedelta}")
        attributes.append(f"latency:{self.latency}")
        attributes.append(f"num_subgraphs:{len(self.child_layers)}")
        attributes.append(f"remaining_depth:{self.depth - self.current_depth}")
        attributes.append(f"num_current_edges:{len(self.edges)}")
        attributes.append(f"caller:{self.caller_node}")

        random.shuffle(attributes)
        return attributes
    
    def get_layer_requirement_description(self, generation_type: TraceGenTaskType, p90_stats, rare_comm_events, rare_event):
        attributes = []
        attributes.append(f"id:{self.service_id}")
        descriptions = []
        if generation_type == TraceGenTaskType.graph_gen_non_recursive:
            if random.choices([True, False], weights=[9, 1])[0]:
                attributes.append(f"num_edges:{self.num_edges}")
            if random.choices([True, False], weights=[9, 1])[0]:
                attributes.append(f"latency:{self.latency}")
            if random.choices([True, False], weights=[9, 1])[0]:
                attributes.append(f"depth:{self.depth}")
            random.shuffle(attributes)
        else:
            special_descriptions = []
            if rare_event:
                src_ms, rpc_type, dest_ms = rare_event
                special_descriptions.append(f"Include an edge from {src_ms} to {dest_ms} with {rpc_type} type")
            else:
                for e in self.edges:
                    if (e.src_ms, e.rpc_type, e.dest_ms) in rare_comm_events:
                        special_descriptions.append(f"Include an edge from {e.src_ms} to {e.dest_ms} with {e.rpc_type} type")
                        break

            if self.start_node == "USER":
                # attributes.append(f"start_node:{self.start_node}")

                if random.choices([True, False], weights=[9, 1])[0]:
                    attributes.append(f"start_node:{self.start_node}")

                if random.choices([True, False], weights=[9, 1])[0]:
                    attributes.append(f"num_edges:{self.num_edges}")
                else:
                    if random.choices([True, False], weights=[9, 1])[0]:
                        descriptions = ["The total number of edges is not provided"]

                if random.choices([True, False], weights=[9, 1])[0]:
                    attributes.append(f"start_edge_id:{self.recent_rpc_id.split('.')[-1]}")
                    if random.choices([True, False], weights=[9, 1])[0]:
                        descriptions.append(start_rpc_id_in_edge_desc(self.recent_rpc_id.split('.')[-1]))

                if random.choices([True, False], weights=[9, 1])[0]:
                    attributes.append(f"start_communication_at:{self.timedelta}")
                    if random.choices([True, False], weights=[9, 1])[0]:
                        descriptions.append(start_communication_at_in_edge_desc(self.timedelta))

                max, p90, avg = p90_stats
                if p90 - avg > 5 and self.latency >= p90:
                    special_descriptions.append("Build a call graph with high latency")
                else:
                    if random.choices([True, False], weights=[5, 5])[0]:
                        attributes.append(f"latency:{self.latency}")
                        if random.choices([True, False], weights=[9, 1])[0]:
                            descriptions.append(edge_start_time_requirement_latency_desc(self.latency))
                        if random.choices([True, False], weights=[9, 1])[0]:
                            descriptions.append(edge_finish_time_requirement_latency_desc(self.latency))
                    else:
                        if random.choices([True, False], weights=[5, 5])[0]:
                            descriptions.append("Build a call graph with latencies based on distributions")

                if random.choices([True, False], weights=[9, 1])[0]:
                    attributes.append(f"num_subgraphs:{len(self.child_layers)}")
                else:
                    if random.choices([True, False], weights=[9, 1])[0]:
                        descriptions.append("Generate subgraph instructions if necessary")
                if random.choices([True, False], weights=[9, 1])[0]:
                    attributes.append(f"remaining_depth:{self.depth - self.current_depth}")
                else:
                    if random.choices([True, False], weights=[9, 1])[0]:
                        descriptions.append("The depth information is not provided")
                if random.choices([True, False], weights=[9, 1])[0]:
                    attributes.append(f"num_current_edges:{len(self.edges)}")
                    if random.choices([True, False], weights=[9, 1])[0]:
                        descriptions.append(num_current_edges_desc(len(self.edges)))
                random.shuffle(attributes)
            else:
                attributes.append(f"num_edges:{self.num_edges}")

                attributes.append(f"start_node:{self.start_node}")

                attributes.append(f"start_edge_id:{self.recent_rpc_id.split('.')[-1]}")
                if random.choices([True, False], weights=[9, 1])[0]:
                    descriptions.append(start_rpc_id_in_edge_desc(self.recent_rpc_id.split('.')[-1]))

                attributes.append(f"start_communication_at:{self.timedelta}")
                if random.choices([True, False], weights=[9, 1])[0]:
                    descriptions.append(start_communication_at_in_edge_desc(self.timedelta))

                attributes.append(f"latency:{self.latency}")
                if random.choices([True, False], weights=[9, 1])[0]:
                    descriptions.append(edge_start_time_requirement_latency_desc(self.latency))
                if random.choices([True, False], weights=[9, 1])[0]:
                    descriptions.append(edge_finish_time_requirement_latency_desc(self.latency))

                attributes.append(f"num_subgraphs:{len(self.child_layers)}")
                if random.choices([True, False], weights=[9, 1])[0]:
                    descriptions.append(num_subgraphs_desc(len(self.child_layers)))
                        
                attributes.append(f"remaining_depth:{self.depth - self.current_depth}")
                attributes.append(f"num_current_edges:{len(self.edges)}")
                if random.choices([True, False], weights=[9, 1])[0]:
                    descriptions.append(num_current_edges_desc(len(self.edges)))
                attributes.append(f"caller:{self.caller_node}")
                random.shuffle(attributes)
        requirements = "/".join(attributes)
        if special_descriptions:
            descriptions += special_descriptions
            # descriptions.append(random.choice(special_descriptions))
        random.shuffle(descriptions)
        return requirements, descriptions


    def to_uncommon_edge_prediction_data_samples(
            self, generation_type: TraceGenTaskType = TraceGenTaskType.graph_gen, rare_event = None, limit: int = 5
        ) -> Tuple[str, bool]:
        output = ""
        cnt = 0
        for e in self.edges:
            if (e.src_ms, e.rpc_type, e.dest_ms) == rare_event:
                break
            output += "(" + e.to_edge_str_with_labels(self.latency, generation_type=generation_type) + ")\n"
            cnt += 1
            if cnt == limit:
                return output, False
        return output, True
    
    def to_prompt_high_latency_prediction(self, target_num_edges: int, generation_type: TraceGenTaskType):
        output = ""
        num_edges = 0
        if not self.merged:
            attributes = [f"start_node:{self.start_node}", f"id:{self.service_id}", f"caller:{self.caller_node}"]
            random.shuffle(attributes)
            output = "/".join(attributes) + "\n"
            output += f"<edges>" + "\n"
            for idx, e in enumerate(self.edges):
                if idx >= target_num_edges:
                    break
                output += "(" + e.to_edge_str_for_high_latency_prediction(generation_type=generation_type) + ")\n"
                num_edges += 1
            output += "</edges>" + "\n"
        return output, num_edges
    
    def to_prompt_infilling_parent(self, target_edge: APICall, generation_type: TraceGenTaskType):
        output = ""
        if generation_type == TraceGenTaskType.graph_gen:
            output += "Predict a missing edge to connect the two layers.\n"
        attributes = [f"start_node:{self.start_node}", f"id:{self.service_id}", f"caller:{self.caller_node}"]
        random.shuffle(attributes)
        output += "/".join(attributes) + "\n"
        output += f"<edges>" + "\n"

        for e in self.edges:
            if e == target_edge:
                edge_str = "[MISSING]"
                output += "(" + edge_str + ")\n"
            else:
                output += "(" + e.to_edge_str_with_labels(latency=0, generation_type=generation_type) + ")\n"
        output += "</edges>" + "\n"
        return output

    def to_prompt_infilling_child(self, generation_type: TraceGenTaskType):
        output = ""
        if generation_type == TraceGenTaskType.graph_gen:
            output += "Child layer:\n"
        attributes = [f"id:{self.service_id}", f"caller:{self.caller_node}"]
        random.shuffle(attributes)
        output += "/".join(attributes) + "\n"
        output += f"<edges>" + "\n"
        for e in self.edges:
            output += "(" + e.to_edge_str_with_labels(latency=0, generation_type=generation_type) + ")\n"
        output += "</edges>" + "\n"
        return output

    def to_prompt_infilling(self, generation_type: TraceGenTaskType):
        output = ""
        ground_truth = []
        if not self.merged:
            if generation_type == TraceGenTaskType.graph_gen:
                output += "Predict a missing element in the following partial call graph.\n"
            attributes = [f"start_node:{self.start_node}", f"id:{self.service_id}", f"caller:{self.caller_node}"]
            random.shuffle(attributes)
            output += "/".join(attributes) + "\n"
            output += f"<edges>" + "\n"
            num_targets = random.randint(1, len(self.edges))
            target_indices = random.sample(range(len(self.edges)), num_targets)

            target_idx = random.randint(0, len(self.edges) - 1)

            for idx, e in enumerate(self.edges):
                if idx == target_idx: 
                    edge_str, missing_data = e.to_edge_str_with_missing_element(generation_type=generation_type)
                    output += "(" + edge_str + ")\n"
                    ground_truth.append(missing_data)
                else:
                    output += "(" + e.to_edge_str_with_labels(latency=0, generation_type=generation_type) + ")\n"
            output += "</edges>" + "\n"
        return output, ground_truth

    def to_prompt_uncommon_edge_prediction(self, target_num_edges: int, generation_type: TraceGenTaskType):
        output = ""
        num_edges = 0
        if not self.merged:
            if generation_type == TraceGenTaskType.graph_gen:
                output += "Predict whether the graph will include an edge not commonly seen given the following partial layers.\n"
            attributes = [f"start_node:{self.start_node}", f"id:{self.service_id}", f"caller:{self.caller_node}"]
            random.shuffle(attributes)
            output += "/".join(attributes) + "\n"
            output += f"<edges>" + "\n"
            for idx, e in enumerate(self.edges):
                if idx >= target_num_edges:
                    break
                output += "(" + e.to_edge_str_with_labels(latency=0, generation_type=generation_type) + ")\n"
                num_edges += 1
            output += "</edges>" + "\n"
        return output, num_edges

    def to_prompt(self, p90_stats, rare_comm_events, generation_type: TraceGenTaskType = TraceGenTaskType.graph_gen, rare_event = None, enable_cot: bool = True) -> dict:
        # requirements = "/".join(self.get_layer_attributes())
        requirements, descriptions = self.get_layer_requirement_description(generation_type, p90_stats, rare_comm_events, rare_event)
        include_description = generation_type == TraceGenTaskType.graph_gen
        if include_description:
            prompt = system_prompt() + "\nRequirements:\n" + requirements + "\n"
        else:
            prompt = requirements + "\n"
        output = ""
        num_edges = self.num_edges
        current_depth = self.current_depth
        remaining_depth = self.depth - current_depth
        if not self.merged:
            invariant_requirements = [
                edge_start_finish_time_desc(),
                subgraph_caller_desc(self.start_node),
            ]
            chosen_requirements = []
            for invariant in invariant_requirements:
                if random.choices([True, False], weights=[9, 1])[0]:
                    chosen_requirements.append(invariant)
            descriptions += chosen_requirements
            random.shuffle(descriptions)
            if include_description:
                prompt += "Conditions:\n" + "\n".join(descriptions) + "\n"
            output += edges_to_prompt(self.edges, self.latency, generation_type) + "\n"

            if include_description and enable_cot:
                edge_max_lat = -1
                max_edge = None
                for e in self.edges:
                    if e.response_time + e.timedelta >= edge_max_lat:
                        edge_max_lat = e.response_time + e.timedelta
                        max_edge = e
                summary = [
                    num_generated_edges_desc(len(self.edges), num_edges, self.edges),
                    # latency_condition_desc(self.latency, max_edge.rpc_id, max_edge.timedelta, max_edge.response_time),
                ]
                output += "\n".join(summary) + "\n"

        if self.child_layers:
            num_remaining_edges = num_edges - (len(self.edges) if not self.merged else 0)
            if self.merged and include_description:
                split_description = [
                    splitted_subgraph_instruction_start_node_desc(self.start_node),
                    split_subgraph_instruction_latency_desc(),
                    split_subgraph_instruction_start_time_desc(),
                ]
                random.shuffle(split_description)
                split_description = [split_subgraph_instructions_desc(5)] + split_description
                prompt += "\n".join(split_description) + "\n"
            if not self.merged and include_description and enable_cot:
                edge_ids = [c.recent_rpc_id.split('.')[0] for c in self.child_layers]
                output += f"generate subgraphs of edge:{',edge_id:'.join(edge_ids)}\n"
            for c in self.child_layers:
                # if self.merged:
                #     subgraph_attributes = c.get_layer_attributes()
                #     output += subgraph_inst(subgraph_attributes, c.recent_rpc_id.split(".")[0]) + "\n"
                # else:
                parent_rpc_id = c.edges[0].rpc_id.rsplit('.', 1)[0]
                parent_edge = None
                for e in c.parent.edges:
                    if e.rpc_id == parent_rpc_id:
                        parent_edge = e
                if include_description and enable_cot:
                    # output += f"start generating subgraph instruction of edge {c.recent_rpc_id.split('.')[0]}\n"
                    output += "Subgraph constraints:\n"
                    conditions = []
                    conditions.append(num_remaining_edges_desc(num_remaining_edges))
                    c_remaining_depth = c.depth - c.current_depth
                    if self.merged:
                        if c_remaining_depth == remaining_depth - 1:
                            conditions.append(splitted_subgraph_instruction_depth_desc(remaining_depth))
                    else:
                        if c_remaining_depth == remaining_depth - 1:
                            conditions.append(subgraph_inst_remaining_depth_necessary_condition_desc(remaining_depth))
                        conditions.append(subgraph_start_node_desc(c.recent_rpc_id.split(".")[0], parent_edge.dest_ms))
                        # conditions.append(subgraph_inst_offset_desc(c.recent_rpc_id.split(".")[0], parent_edge.timedelta))
                        # conditions.append(subgraph_inst_latency_desc(parent_rpc_id, parent_edge.timedelta, parent_edge.response_time))
                    random.shuffle(conditions)
                    output += "\n".join(conditions) + "\n"
                subgraph_attributes = c.get_layer_attributes()
                output += subgraph_inst(subgraph_attributes, c.recent_rpc_id.split(".")[0]) + "\n"
                if include_description and enable_cot:
                    output += generated_subgraph_inst_desc(num_remaining_edges, c.num_edges) + "\n"
                num_remaining_edges -= c.num_edges
                if num_remaining_edges == 0 and include_description and enable_cot:
                    output += "finish generation"

        return {"instruction": prompt, "output": output}
