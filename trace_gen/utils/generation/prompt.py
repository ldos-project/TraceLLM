from typing import List
from trace_gen.schema.api_call import APICall, APICallAttributes
from trace_gen.generate.trace_oracle.schema.task_type import TraceGenTaskType

def system_prompt() -> str:
    return "You are a trace generator that creates traces based on given requirements."

# Edge trace format
def edges_to_prompt(edges: List[APICall], latency: int, generation_type: TraceGenTaskType) -> str:
    edge_prompt = ""
    edge_prompt += f"<edges>" + "\n"
    for e in edges:
        edge_prompt += "(" + e.to_edge_str_with_labels(latency, generation_type=generation_type) + ")\n"
    edge_prompt += "</edges>"
    return edge_prompt

# Current layer description
def current_depth_desc(start_rpc_id: str, current_depth: int) -> str:
    return f"current_depth = 1 + the number of . in start_rpc_id = 1 + '{start_rpc_id}'.count('.') = {current_depth}"

def remaining_depth_desc(depth: int, current_depth: int, remaining_depth: int) -> str:
    return f"remaining_depth = max_depth - current_depth = {depth} - {current_depth} = {remaining_depth}"

def max_num_edges_desc(num_edges: int, remaining_depth: int) -> str:
    return f"maximum number of edges = num_edges - remaining_depth = {num_edges} - {remaining_depth} = {num_edges - remaining_depth}"

def num_current_edges_desc(num_edges_in_current_layer: int) -> str:
    return f"generate {num_edges_in_current_layer} edges following num_current_edges"

def generate_one_edge_desc() -> str:
    return "should generate only 1 edge"

def positive_remaining_depth_desc(remaining_depth: int, num_available_edges: int) -> str:
    return f"remaining_depth is {remaining_depth} > 0 => generate less than {num_available_edges} edges"

def zero_remaining_depth_desc(num_available_edges: int) -> str:
    return f"remaining_depth is 0 => no more subgraphs, generate {num_available_edges} edges"

def start_rpc_id_in_edge_desc(start_rpc_id: str) -> str:
    return f"the first edge_id should be start_edge_id {start_rpc_id}"

def start_communication_at_in_edge_desc(offset: int) -> str:
    return f"the first start_communication_at should be requirement's start_communication_at {offset}"

# Subgraph generation description
def num_generated_edges_desc(generated: int, total: int, edges: list[APICall]) -> str:
    desc = f"num generated edges = the last edge id - the first edge id + 1 = {edges[-1].recent_rpc_id} - {edges[0].recent_rpc_id} + 1 = {generated}\n"
    desc += f"{generated} edges generated out of num_edges:{total}\n"
    desc += f"num_remaining_edges = num_edges:{total} - generated:{generated} = {total - generated}\n"
    if total - generated == 0:
        desc += "finish generation"
    return desc

def num_subgraphs_desc(num_remaining_subgraphs: int) -> str:
    if num_remaining_subgraphs == 0:
        return "no subgraph to generate"
    return f"generate {num_remaining_subgraphs} number of subgraphs"

def latency_condition_desc(latency: int, rpc_id: str, timedelta: int, response_time: int) -> str:
    """Total latency number should be equal to or larger than the latency of the maximum value in edges."""
    return f"latency {latency} >= offset + response_time of edge with rpc_id({rpc_id}) = {timedelta} + {response_time}"

def num_remaining_edges_desc(num_remaining_edges: int) -> str:
    return f"num_edges <= num_remaining_edges:{num_remaining_edges}"

def subgraph_inst_latency_desc(parent_rpc_id: str, offset: int, response_time: int) -> str:
    return f"latency <= finish time of edge {parent_rpc_id.split('.')[-1]} = {offset + response_time}"

def subgraph_inst_remaining_depth_necessary_condition_desc(remaining_depth: int) -> str:
    return f"remaining_depth should be the requirement's remaining_depth:{remaining_depth} - 1 = {remaining_depth - 1}"

def subgraph_inst_remaining_depth_desc(remaining_depth: int) -> str:
    return f"remaining_depth < the current layer's remaining_depth {remaining_depth}"

def subgraph_inst_offset_desc(edge_id:int, offset: int) -> str:
    return f"start_communication_at >= edge {edge_id} communication start time: {offset}"

def subgraph_start_node_desc(edge_id:int, start_node: str) -> str:
    return f"copy start_node from edge {edge_id} destination: {start_node}"

def subgraph_caller_desc(start_node: str) -> str:
    return f"copy caller from requirement's start_node:{start_node}"

def subgraph_inst(subgraph_attributes: List[str], index: int) -> str:
    return f"<subgraph of edge_id {index}>" + "/".join(subgraph_attributes) + "</subgraph>"

def generated_subgraph_inst_desc(num_remaining_edges: int, num_generated: int) -> str:
    return f"now, num_remaining_edges is {num_remaining_edges} - {num_generated} = {num_remaining_edges - num_generated}"

def edge_start_finish_time_desc() -> str:
    return "for all edges, communication start time <= communication finish time"

def edge_start_time_latency_desc() -> str:
    return "for all edges, communication start time <= requirement's latency"

def edge_start_time_requirement_latency_desc(latency: int) -> str:
    return f"In each edge, communication start time should NOT be greater than latency {latency} milliseconds"

def edge_finish_time_requirement_latency_desc(latency: int) -> str:
    return f"Also, communication should finish before latency {latency} milliseconds"

def edge_finish_time_latency_desc() -> str:
    return "for all edges, communication finish time <= requirement's latency"

### Middle Layer ###
def split_subgraph_instructions_desc(edge_threshold: int = 15) -> str:
    return f"num_current_edges is over the threshold, so split the subgraph instructions into smaller subgraph instructions"

def splitted_subgraph_instruction_start_node_desc(start_node: str) -> str:
    return f"When splitting the subgraph instructions, copy start_node:{start_node} from the requirement"

def splitted_subgraph_instruction_depth_desc(remaining_depth: int) -> str:
    return f"copy 'remaining_depth:{remaining_depth}' from the requirement"  

def split_subgraph_instruction_start_time_desc() -> str:
    return "for all subgraph instructions, start_communication_at <= latency"

def split_subgraph_instruction_latency_desc() -> str:
    return "for all subgraph instructions, latency should not be larger than the requirement's latency"

