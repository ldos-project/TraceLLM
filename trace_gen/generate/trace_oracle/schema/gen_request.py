from __future__ import annotations

from typing import List
from itertools import chain

import random
from pydantic import BaseModel
from trace_gen.generate.trace_oracle.schema.task_type import TraceGenTaskType, GraphGenInstruction
from trace_gen.generate.trace_oracle.schema.validator import GenOutputValidator
from trace_gen.generate.trace_oracle.schema.errors import GenerationError
from trace_gen.schema.api_call import APICall
from trace_gen.utils.generation.prompt import (
    start_rpc_id_in_edge_desc,
    num_current_edges_desc,
    start_communication_at_in_edge_desc,
    split_subgraph_instructions_desc,
    splitted_subgraph_instruction_start_node_desc,
    split_subgraph_instruction_latency_desc,
    split_subgraph_instruction_start_time_desc,
    system_prompt,
    edge_start_finish_time_desc,
    edge_start_time_requirement_latency_desc,
    edge_finish_time_requirement_latency_desc,
)


from vllm import SamplingParams


class GenRequest(BaseModel):
    id: int = -1
    prompt: str
    task_type: TraceGenTaskType
    sampling_params: dict
    trials: List[str] = []
    invalid_reasons: List[List[GenerationError]] = []
    child_requests: List[GenRequest] = []
    is_valid: bool = False
    is_child: bool = False
    generated_edges: List[APICall] = []
    rpc_id_prefix: str = ""
    svc_id: str = "S_000000000"

    @property
    def num_trials(self):
        return len(self.trials)

    @property
    def instruction(self):
        return GraphGenInstruction.from_prompt(self.prompt.strip("[GENERATE GRAPH]").split("\n", 1)[0].rstrip("["))

    @property
    def is_finished(self):
        if self.is_valid and all([c.is_finished for c in self.child_requests]):
            return True
        return False

    def is_ongoing(self, num_trials: int):
        if not self.is_valid:
            if len(self.trials) < num_trials:
                return True
            return False
        else:
            return any([c.is_ongoing(num_trials=num_trials) for c in self.child_requests])

    def check_validity(self, validator: GenOutputValidator) -> bool:
        if self.trials:
            last_trial = self.trials[-1]
            invalid_reason = validator.validate(self.task_type, self.prompt + last_trial)
            self.invalid_reasons.append(invalid_reason)
            self.is_valid = False if invalid_reason else True
        else:
            self.is_valid = False

    def make_child_requests(self, validator: GenOutputValidator, include_description=True):
        edge_threshold = 15
        if self.is_valid:
            last_trial = self.trials[-1]
            parsed = validator.parse(self.task_type, self.prompt + last_trial)
            if self.task_type == TraceGenTaskType.graph_gen or self.task_type == TraceGenTaskType.graph_gen_recursive:
                instruction, api_calls, subgraph_instructions = parsed
                self.generated_edges = api_calls
                for (edge_idx, subgraph_instruction) in subgraph_instructions:
                    sg_gen_instruction = GraphGenInstruction.from_prompt(subgraph_instruction)
                    if self.task_type ==TraceGenTaskType.graph_gen:
                        prompt = "### Question: " + system_prompt() + "\nRequirements:\n" + subgraph_instruction + "\n"
                    else:
                        prompt = subgraph_instruction + "\n"
                    if include_description:
                        conditions_to_propagate = []
                        for prompt_line in self.prompt.split("\n"):
                            if prompt_line.startswith("Include an edge from"):
                                conditions_to_propagate.append(prompt_line)
                        if sg_gen_instruction.num_current_edges > edge_threshold:
                            description = [
                                splitted_subgraph_instruction_start_node_desc(sg_gen_instruction.start_node),
                                split_subgraph_instruction_latency_desc(),
                                split_subgraph_instruction_start_time_desc(),
                            ]
                            description += conditions_to_propagate
                            random.shuffle(description)
                            description = [split_subgraph_instructions_desc(5)] + description
                            prompt += "\n".join(description) + "\n ### Answer: \n<split>\n"
                        else:
                            description = [
                                num_current_edges_desc(sg_gen_instruction.num_current_edges),
                                start_rpc_id_in_edge_desc(sg_gen_instruction.start_rpc_id),
                                start_communication_at_in_edge_desc(sg_gen_instruction.offset),
                                edge_start_finish_time_desc(),
                                edge_start_time_requirement_latency_desc(sg_gen_instruction.latency),
                                edge_finish_time_requirement_latency_desc(sg_gen_instruction.latency),
                            ]
                            description += conditions_to_propagate
                            random.shuffle(description)
                            prompt += "Conditions:\n" + "\n".join(description) + "\n ### Answer: \n<layer>\n<edges>"
                    if self.generated_edges:
                        rpc_id_prefix = f"{self.rpc_id_prefix}.{edge_idx}" if self.rpc_id_prefix else str(edge_idx)
                    else:
                        rpc_id_prefix = self.rpc_id_prefix

                    tag = "<split>" if sg_gen_instruction.num_current_edges > edge_threshold else "<layer>"
                    self.child_requests.append(
                        GenRequest(
                            id=self.id,
                            prompt=prompt,
                            task_type=self.task_type,
                            sampling_params=self.sampling_params,
                            trials=[],
                            child_requests=[],
                            is_child=True,
                            rpc_id_prefix=rpc_id_prefix,
                        )
                    )

    def get_payload(self):
        return {
            "prompt": self.prompt,
            **self.sampling_params,
        }

    def collected_generated_edges(self) -> List[APICall]:
        current_layer_edges = []
        for edge in self.generated_edges:
            modifed_edge = edge
            if self.rpc_id_prefix:
                modifed_edge.rpc_id = f"{self.rpc_id_prefix}.{edge.rpc_id}"
            current_layer_edges.append(modifed_edge)
        return current_layer_edges + list(chain(*[c.collected_generated_edges() for c in self.child_requests]))
