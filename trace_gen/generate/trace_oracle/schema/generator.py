import asyncio
import json
from collections import deque
from typing import List
import httpx

from trace_gen.generate.trace_oracle.schema.gen_request import GenRequest
from trace_gen.generate.trace_oracle.schema.task_type import TraceGenTaskType
from trace_gen.generate.trace_oracle.schema.validator import GenOutputValidator
from trace_gen.schema.call_graph import CallGraph

class TraceGenerator:

    def __init__(
        self, endpoint: str, validator: GenOutputValidator, max_batch_size: int = 100
    ):
        self.endpoint = endpoint
        self.max_batch_size = max_batch_size
        self.validator = validator

    async def start_generation(
        self,
        requests: list[GenRequest],
        output_path: str,
        failure_log: str,
        root_requests: List[GenRequest] = []
    ):
        headers = {"User-Agent": "Test Client"}
        request_queue: deque[GenRequest] = deque()
        for request in requests:
            request_queue.append(request)

        async with httpx.AsyncClient(base_url=self.endpoint) as client:
            current_batch: List[GenRequest] = []
            while (
                len(request_queue) and len(current_batch) < self.max_batch_size
            ):
                current_batch.append(request_queue.popleft())

            outputs = await asyncio.gather(
                *[
                    client.post(
                        "/generate",
                        headers=headers,
                        json=item.get_payload(),
                        timeout=1200.0,
                    )
                    for item in current_batch
                ]
            )

            responses = [json.loads(response.content) for response in outputs]

            # append outputs to GenRequests
            num_retry = 1
            with open(output_path, "a") as f:
                with open(failure_log, "a") as e:
                    with open(failure_log + ".error_code", "a") as ec:
                        for request, response in zip(current_batch, responses):
                            generated_text = response["text"][0]
                            request.trials.append(generated_text)
                            request.check_validity(self.validator)

                            if request.is_valid:
                                if request.task_type == TraceGenTaskType.graph_gen or TraceGenTaskType.graph_gen_recursive:
                                    request.make_child_requests(self.validator, include_description=True if request.task_type == TraceGenTaskType.graph_gen else False)
                                    for child_request in request.child_requests:
                                        request_queue.appendleft(child_request)
                                if not request.is_child:
                                    root_requests.append(request)

                            else:
                                invalid_reasons = request.invalid_reasons[-1]
                                error_codes = ",".join([str(reason.error_code) for reason in invalid_reasons])
                                ec.write(f"<{request.id}>" + error_codes + "\n")
                                error_msg = '/'.join([reason.detail for reason in invalid_reasons])
                                e.write(f"<{request.id}>FAILED CASE\nReasons: {error_msg}\nResult:\n")
                                e.write(f"{request.trials[-1]}\n")
                                if len(request.trials) < num_retry:
                                    request_queue.appendleft(request)
                                else:
                                    if not request.is_child:
                                        root_requests.append(request)

        unfinished_requests = []
        num_finished = 0
        num_valid = 0
        num_correct_instruction = 0
        sum_depth = 0
        sum_edges = 0
        with open(output_path, "a") as f:
            for request in root_requests:
                if request.is_finished:
                    num_finished += 1
                    if request.task_type == TraceGenTaskType.graph_gen or request.task_type == TraceGenTaskType.graph_gen_recursive:
                        cg = CallGraph(
                            service_id=int(request.svc_id.strip("S_")),
                            edges=request.collected_generated_edges(),
                        )
                        if cg.valid:
                            num_valid += 1
                            correct = True
                            if request.instruction.num_edges > 0 and len(cg.edges) != request.instruction.num_edges:
                                correct = False
                            if request.instruction.depth >= 0 and cg.edge_dag.root.depth != request.instruction.depth + 1:
                                correct = False
                            if correct:
                                num_correct_instruction += 1
                            sum_depth += cg.edge_dag.root.depth
                            sum_edges += len(cg.edges)

                            synthetic = f"<s> id:S_{str(cg.service_id).zfill(9)}/max depth:{cg.edge_dag.root.depth}/num edges:{len(cg.edges)}"
                            synthetic += "[" + "||".join([e.to_edge_str_full_info() for e in cg.edges]) + "]</s>"
                            f.write(f"<{request.id}>" + synthetic + "\n")
                        else:
                            f.write("[INVALID TRACE]" + ",".join(cg.invalid_reason))
                            synthetic = "[" + "||".join([e.to_edge_str_full_info() for e in cg.edges]) + "]</s>"
                            f.write(synthetic + "\n")
                    elif request.task_type == TraceGenTaskType.graph_gen_non_recursive:
                        num_valid += 1
                        instruction, cg = self.validator.parse(request.task_type, request.trials[-1])
                        correct = True
                        if request.instruction.num_edges > 0 and len(cg.edges) != instruction.num_edges:
                            correct = False
                        if request.instruction.depth >= 0 and cg.edge_dag.root.depth != instruction.depth:
                            correct = False
                        if correct:
                            num_correct_instruction += 1
                            synthetic = f"<s> id:S_{str(cg.service_id).zfill(9)}/max depth:{cg.edge_dag.root.depth}/num edges:{len(cg.edges)}"
                            synthetic += "[" + "||".join([e.to_edge_str_full_info() for e in cg.edges]) + "]</s>"
                            f.write(f"<{request.id}>" + synthetic + "\n")

                        sum_depth += cg.edge_dag.root.depth
                        sum_edges += len(cg.edges)
                else:
                    if request.is_ongoing(num_retry):
                        unfinished_requests.append(request)

        return list(request_queue), unfinished_requests, (num_finished, num_valid, num_correct_instruction, sum_edges, sum_depth)
