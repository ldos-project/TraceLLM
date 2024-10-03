from trace_gen.generate.trace_oracle.schema.errors import (
    GenerationError,
    InvalidCallGraph,
    InvalidInstruction,
    InvalidNumEdges,
    InvalidMaxDepth,
    LargeLatencyInEdges,
    LargeLatencyInSubgraphs,
    LargeOffsetInSubgraphInst,
    MoreEdgesGenerated,
    NoSubgraphInst,
    ParsingException,
    WrongEdgeStartRPCID,
    WrongMaxDepthInSubgraphs,
    WrongSubgraphStartNode,
    SubgraphWithLessDepth,
    SubgraphDepthNotSameWhileSplitting,
    EdgesNotSplitted,
)
from trace_gen.generate.trace_oracle.schema.task_type import (
    GraphGenInstruction,
    TraceGenTaskType,
)
from trace_gen.schema.api_call import APICall, APICallAttributes
from trace_gen.schema.call_graph import CallGraph


class GenOutputValidator:

    def parse(self, task_type: TraceGenTaskType, prompt_output: str):
        if task_type == TraceGenTaskType.graph_gen_recursive or task_type == TraceGenTaskType.graph_gen:
            if task_type == TraceGenTaskType.graph_gen_recursive:
                instruction, edges_str = prompt_output.split("\n", 1)
                instruction = GraphGenInstruction.from_prompt(instruction.strip())
            else:
                # valid if output has correct format
                instruction, edges_str = prompt_output.split("### Answer:", 1)
                _, requirements = instruction.split("Requirements:", 1)
                requirements = requirements.strip().split("\n", 1)
                instruction = GraphGenInstruction.from_prompt(requirements[0])

            edges = []
            api_calls = []
            if "<edges>" in edges_str:
                edges_str = edges_str.split("<edges>", 1)[-1]
                # edges_str = edges_str.split(">", 1)[-1]
                edges_str_split = edges_str.split("</edges>", 1)
                edges_str = edges_str_split[0]
                if len(edges_str) > 1:
                    remaining = edges_str_split[-1]
                else:
                    remaining = ""
                edges = [e.strip("(").rstrip(")") for e in edges_str.split("\n") if e.startswith("(") and e.endswith(")")]
            else:
                remaining = edges_str
            for edge in edges:
                for edge_attribute in edge.split(","):
                    if APICallAttributes.EDGE_ID.value in edge_attribute:
                        edge_idx = edge_attribute.split(" is ")[-1]
                    elif APICallAttributes.RPC_TYPE.value in edge_attribute:
                        rpc_type = edge_attribute.split(" is ")[-1]
                    elif APICallAttributes.DEST.value in edge_attribute:
                        dest_ms = edge_attribute.split(" is ")[-1]
                    elif APICallAttributes.OFFSET.value in edge_attribute:
                        timedelta = int(edge_attribute.split(" at ")[-1].rstrip(" milliseconds"))
                    elif APICallAttributes.FINISHED_AT.value in edge_attribute:
                        finished_at = int(edge_attribute.split(" at ")[-1].rstrip(" milliseconds"))
                rt = finished_at - timedelta
                # edge_idx, rpc_type, dest_ms, timedelta, rt = edge.split(",")
                api_calls.append(
                    APICall(
                        rpc_id=edge_idx,
                        timedelta=timedelta,
                        rpc_type=rpc_type,
                        src_ms=instruction.start_node,
                        dest_ms=dest_ms,
                        response_time=rt,
                    )
                )

            subgraph_instructions = []
            if "<subgraph of edge_id " in remaining:
                summary, subgraph_str = remaining.split("<subgraph of edge_id ", 1)
                subgraph_str, subgraph_summary = subgraph_str.rsplit("</subgraph>", 1)
                subgraph_str = subgraph_str.strip("\n")
                subgraph_prompts = subgraph_str.split("</subgraph>")
                subgraph_instructions = []
                for subgraph_prompt in subgraph_prompts:
                    edge_idx, prompt = subgraph_prompt.rsplit(">", 1)
                    if "<subgraph of edge_id " in edge_idx:
                        edge_idx = edge_idx.split("<subgraph of edge_id ")[-1]
                    subgraph_instructions.append((edge_idx, prompt))
            return (instruction, api_calls, subgraph_instructions)

        if task_type == TraceGenTaskType.graph_gen_non_recursive:
            prompt_output = prompt_output.rstrip("</s>")
            instruction = prompt_output.split("<edges>", 1)[0]
            instruction = GraphGenInstruction.from_prompt(
                instruction.strip("[GENERATE GRAPH]").split("\n", 1)[0]
            )
            edges_str = prompt_output.split("<edges>", 1)[-1].split("\n")
            api_calls = []
            edges = [e.strip("(").rstrip(")") for e in edges_str if e.startswith("(") and e.endswith(")")]
            for edge in edges:
                for edge_attribute in edge.split(","):
                    if APICallAttributes.EDGE_ID.value in edge_attribute:
                        edge_idx = edge_attribute.split(" is ")[-1]
                    elif APICallAttributes.RPC_TYPE.value in edge_attribute:
                        rpc_type = edge_attribute.split(" is ")[-1]
                    elif APICallAttributes.DEST.value in edge_attribute:
                        dest_ms = edge_attribute.split(" is ")[-1]
                    elif APICallAttributes.SRC.value in edge_attribute:
                        src_ms = edge_attribute.split(" is ")[-1]
                    elif APICallAttributes.OFFSET.value in edge_attribute:
                        timedelta = int(edge_attribute.split(" at ")[-1].rstrip(" milliseconds"))
                    elif APICallAttributes.FINISHED_AT.value in edge_attribute:
                        finished_at = int(edge_attribute.split(" at ")[-1].rstrip(" milliseconds"))
                rt = finished_at - timedelta
                api_calls.append(
                    APICall(
                        rpc_id=edge_idx,
                        timedelta=timedelta,
                        rpc_type=rpc_type,
                        src_ms=src_ms,
                        dest_ms=dest_ms,
                        response_time=rt,
                    )
                )
            cg = CallGraph.get_call_graph_from_trace(
                f"00h00m00s000ms {instruction.id} " + '||'.join([f"{api_call.to_edge_str_full_info()}" for api_call in api_calls])
            )
            return instruction, cg

        if task_type == TraceGenTaskType.context_gen:
            pass

    def validate(
        self, task_type: TraceGenTaskType, prompt_output: str
    ) -> list[GenerationError]:
        invalid_reasons = []
        try:
            parsed = self.parse(task_type, prompt_output)
            if task_type == TraceGenTaskType.graph_gen_non_recursive:
                instruction, cg = parsed
                invalid_reasons = []
                if cg.valid:
                    if instruction.num_edges >= 0 and len(cg.edges) != instruction.num_edges:
                        invalid_reasons.append(
                            InvalidNumEdges.from_args(
                                desired_num_edges=instruction.num_edges,
                                current_num_edges=len(cg.edges),
                            )
                        )
                    if instruction.depth >= 0 and cg.edge_dag.root.depth != instruction.depth:
                        invalid_reasons.append(
                            InvalidMaxDepth.from_args(
                                desired_depth=instruction.depth,
                                current_depth=cg.edge_dag.root.depth,
                            )
                        )
                else:
                    invalid_reasons.append(InvalidCallGraph.from_args(cg.invalid_reason))
                return invalid_reasons
            if task_type == TraceGenTaskType.graph_gen or task_type == TraceGenTaskType.graph_gen_recursive:
                instruction, api_calls, subgraph_instructions = parsed
                subgraph_gen_instructions = [
                    (edge_idx, GraphGenInstruction.from_prompt(subgraph_instruction))
                    for (edge_idx, subgraph_instruction) in subgraph_instructions
                ]
                if not instruction.valid or not all(
                    [
                        subgraph_gen_instruction.valid
                        for (_, subgraph_gen_instruction) in subgraph_gen_instructions
                    ]
                ):
                    for (_, subgraph_inst) in subgraph_gen_instructions:
                        if not subgraph_inst.valid:
                            invalid_reasons.append(
                                InvalidInstruction.from_args(instruction=subgraph_inst)
                            )

                num_subgraph_edges = sum(
                    [
                        subgraph_gen_instruction.num_edges
                        for (_, subgraph_gen_instruction) in subgraph_gen_instructions
                    ]
                )
                if instruction.num_edges > 0 and instruction.num_edges != len(api_calls) + num_subgraph_edges:
                    invalid_reasons.append(
                        InvalidNumEdges.from_args(
                            desired_num_edges=instruction.num_edges,
                            current_num_edges=len(api_calls) + num_subgraph_edges,
                        )
                    )

                if subgraph_gen_instructions:
                    for (edge_idx, sg_gen_inst) in subgraph_gen_instructions:
                        e = None
                        for edge in api_calls:
                            if edge.rpc_id == edge_idx:
                                e = edge
                                break
                        if not e and api_calls:
                            invalid_reasons.append(
                                WrongEdgeStartRPCID.from_args(
                                    desired=edge_idx,
                                    current=",".join([edge.rpc_id for edge in api_calls]),
                                )
                            )
                        if e and e.dest_ms != sg_gen_inst.start_node:
                            invalid_reasons.append(
                                WrongSubgraphStartNode.from_args(
                                    desired=e.dest_ms,
                                    current=sg_gen_inst.start_node,
                                )
                            )
                        if sg_gen_inst.offset > sg_gen_inst.latency:
                            invalid_reasons.append(
                                LargeOffsetInSubgraphInst.from_args(
                                    offset=sg_gen_inst.offset,
                                    latency=sg_gen_inst.latency,
                                )
                            )
                        if instruction.latency >= 0 and instruction.latency < sg_gen_inst.latency:
                            invalid_reasons.append(
                                LargeLatencyInSubgraphs.from_args(
                                    parent_latency=instruction.latency,
                                    child_latency=sg_gen_inst.latency,
                                )
                            )
                        if e and (e.timedelta + e.response_time < sg_gen_inst.latency):
                            invalid_reasons.append(
                                LargeLatencyInSubgraphs.from_args(
                                    parent_latency=e.timedelta + e.response_time,
                                    child_latency=sg_gen_inst.latency,
                                )
                            )
                        if instruction.depth >= 0 and sg_gen_inst.depth > instruction.depth:
                            invalid_reasons.append(
                                WrongMaxDepthInSubgraphs.from_args(
                                    parent_max_depth=instruction.depth,
                                    child_max_depth=sg_gen_inst.depth,
                                )
                            )
                    # At least one subgraph should match parent's depth
                    if api_calls:
                        if instruction.depth >= 0 and not all(
                            [
                                sg_gen_inst.depth < instruction.depth
                                for (_, sg_gen_inst) in subgraph_gen_instructions
                            ]
                        ):
                            invalid_reasons.append(
                                SubgraphWithLessDepth.from_args(
                                    instruction_depth=instruction.depth,
                                    subgraph_depths=[s.depth for (_, s) in subgraph_gen_instructions],
                                )
                            )
                        if instruction.depth >= 0 and not any(
                        [
                            sg_gen_inst.depth == instruction.depth - 1
                            for (_, sg_gen_inst) in subgraph_gen_instructions
                        ]
                        ):
                            invalid_reasons.append(
                                SubgraphWithLessDepth.from_args(
                                    instruction_depth=instruction.depth,
                                    subgraph_depths=[s.depth for (_, s) in subgraph_gen_instructions],
                                )
                            )
                    else:
                        if instruction.depth >= 0 and not any(
                            [
                                sg_gen_inst.depth == instruction.depth
                                for (_, sg_gen_inst) in subgraph_gen_instructions
                            ]
                        ):
                            invalid_reasons.append(
                                SubgraphDepthNotSameWhileSplitting.from_args(
                                    instruction_depth=instruction.depth,
                                    subgraph_depths=[s.depth for (_, s) in subgraph_gen_instructions],
                                )
                            )
                        if not all([sg_gen_inst.num_edges < instruction.num_edges for (_, sg_gen_inst) in subgraph_gen_instructions]):
                            invalid_reasons.append(
                                EdgesNotSplitted.from_args(
                                    desired_edges=instruction.num_edges,
                                    num_generated_edges=sum([sg_gen_inst.num_edges for (_, sg_gen_inst) in subgraph_gen_instructions]),
                                )
                            )

                if (
                    not subgraph_gen_instructions
                    and instruction.depth > 0
                ):
                    if instruction.num_edges > 0 and instruction.num_edges > len(api_calls):
                        invalid_reasons.append(
                            NoSubgraphInst.from_args(
                                desired_depth=instruction.depth, current_depth=-1
                            )
                        )

                if api_calls:
                    max_latency = max([e.timedelta + e.response_time for e in api_calls])
                    if instruction.latency >= 0 and instruction.latency < max_latency:
                        invalid_reasons.append(
                            LargeLatencyInEdges.from_args(
                                desired_latency=instruction.latency,
                                generated_latency=max_latency,
                            )
                        )

                    if instruction.num_edges > 0 and instruction.depth >= 0 and instruction.num_edges - instruction.depth == 1:
                        if len(api_calls) != 1:
                            invalid_reasons.append(
                                MoreEdgesGenerated.from_args(
                                    desired_edges=instruction.num_edges,
                                    remaining_depth=instruction.depth,
                                    num_generated_edges=len(api_calls),
                                )
                            )
            return invalid_reasons
        except Exception as e:
            invalid_reasons.append(ParsingException.from_args(exception_error=str(e)))
            return invalid_reasons
