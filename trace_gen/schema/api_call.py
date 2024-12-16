from __future__ import annotations

from typing import List

from parse import parse
from pydantic import BaseModel
import re
from enum import Enum
import random
from packaging.version import Version
from trace_gen.generate.trace_oracle.schema.task_type import TraceGenTaskType

def ms_repl_fn(m):
  return f'MS_{m.group(1).zfill(5)}'

class APICallAttributes(Enum):
    EDGE_ID = "edge_id"
    RPC_TYPE = "type"
    SRC = "source"
    DEST = "destination"
    OFFSET = "communication starts at"
    RESPONSE_TIME = "response_time"
    FINISHED_AT = "communication finishes at"
    LATENCY_REQUIREMENT = "should finish before latency"

class APICall(BaseModel):
    rpc_id: str
    timedelta: int
    rpc_type: str
    src_ms: str
    dest_ms: str
    response_time: int
    invalid_reason: List[str] = []

    def __lt__(self, other: APICall):
        if(Version(self.rpc_id) < Version(other.rpc_id)):
            return True
        return False

    def __hash__(self):
        return abs(
            hash(",".join([self.rpc_id, self.rpc_type, self.src_ms, self.dest_ms, str(self.timedelta), str(self.response_time)]))
        ) % (10**10)

    def hash_wo_rpc_id(self):
        return abs(
            hash(",".join([self.rpc_type, self.src_ms, self.dest_ms]))
        ) % (10**10)
    
    def get_parent_rpc_id(self):
        if "." not in self.rpc_type:
            return None
        return self.rpc_type.rsplit(".")[0]
    
    @property
    def depth(self):
        return self.rpc_id.count(".") + 1

    @property
    def recent_rpc_id(self):
        return '.'.join(self.rpc_id.rsplit('.', 1)[-1:])
    
    @property
    def finish_time(self):
        return self.timedelta + self.response_time

    def to_edge_str(self):
        return "|".join(
            [self.recent_rpc_id, self.rpc_type, self.dest_ms, "ARRIVED_" + str(self.timedelta), "RT_" + str(self.response_time)]
        )

    def to_edge_str_start_finish_time(self):
        edge_attributes = [
                f"{APICallAttributes.EDGE_ID.value} is {self.recent_rpc_id}",
                f"{APICallAttributes.RPC_TYPE.value} is {self.rpc_type}",
                f"{APICallAttributes.DEST.value} is {self.dest_ms}",
                f"{APICallAttributes.OFFSET.value} {str(self.timedelta)} milliseconds",
                f"{APICallAttributes.FINISHED_AT.value} {str(self.timedelta + self.response_time)} milliseconds",
            ]
        random.shuffle(edge_attributes)
        return ",".join(edge_attributes) 

    def to_edge_str_for_high_latency_prediction(self, generation_type: TraceGenTaskType):
        if generation_type == TraceGenTaskType.graph_gen_non_recursive:
            edge_attributes = [
                f"{APICallAttributes.EDGE_ID.value} is {self.rpc_id}",
                f"{APICallAttributes.RPC_TYPE.value} is {self.rpc_type}",
                f"{APICallAttributes.SRC.value} is {self.src_ms}",
                f"{APICallAttributes.DEST.value} is {self.dest_ms}",
                f"{APICallAttributes.OFFSET.value} {str(self.timedelta)} milliseconds",
            ]
        else:
            edge_attributes = [
                f"{APICallAttributes.EDGE_ID.value} is {self.recent_rpc_id}",
                f"{APICallAttributes.RPC_TYPE.value} is {self.rpc_type}",
                f"{APICallAttributes.DEST.value} is {self.dest_ms}",
                f"{APICallAttributes.OFFSET.value} {str(self.timedelta)} milliseconds",
            ]
        random.shuffle(edge_attributes)
        return ",".join(edge_attributes)
    
    def to_edge_str_with_missing_element(self, generation_type: TraceGenTaskType):
        target_attribute = random.choice(
            [
                APICallAttributes.RPC_TYPE,
                APICallAttributes.DEST,
            ]
        )
        if generation_type != TraceGenTaskType.graph_gen_non_recursive:
            edge_attributes = [
                f"{APICallAttributes.EDGE_ID.value} is {self.recent_rpc_id}",
                f"{APICallAttributes.OFFSET.value} {str(self.timedelta)} milliseconds",
                f"{APICallAttributes.FINISHED_AT.value} {str(self.timedelta + self.response_time)} milliseconds",
            ]
            if target_attribute == APICallAttributes.RPC_TYPE:
                ground_truth = self.rpc_type
                edge_attributes.append(f"{APICallAttributes.RPC_TYPE.value} is [MISSING]")
            else:
                edge_attributes.append(f"{APICallAttributes.RPC_TYPE.value} is {self.rpc_type}")
            if target_attribute == APICallAttributes.DEST:
                ground_truth = self.dest_ms
                edge_attributes.append(f"{APICallAttributes.DEST.value} is [MISSING]")
            else:
                edge_attributes.append(f"{APICallAttributes.DEST.value} is {self.dest_ms}")
            
        random.shuffle(edge_attributes)
        return ",".join(edge_attributes), ground_truth


    def to_edge_str_with_labels(self, latency: int, generation_type: TraceGenTaskType):
        if generation_type == TraceGenTaskType.graph_gen_non_recursive:
            edge_attributes = [
                f"{APICallAttributes.EDGE_ID.value} is {self.rpc_id}",
                f"{APICallAttributes.RPC_TYPE.value} is {self.rpc_type}",
                f"{APICallAttributes.SRC.value} is {self.src_ms}",
                f"{APICallAttributes.DEST.value} is {self.dest_ms}",
                f"{APICallAttributes.OFFSET.value} {str(self.timedelta)} milliseconds",
                f"{APICallAttributes.FINISHED_AT.value} {str(self.timedelta + self.response_time)} milliseconds",
            ]
        else:
            edge_attributes = [
                f"{APICallAttributes.EDGE_ID.value} is {self.recent_rpc_id}",
                f"{APICallAttributes.RPC_TYPE.value} is {self.rpc_type}",
                f"{APICallAttributes.DEST.value} is {self.dest_ms}",
                f"{APICallAttributes.OFFSET.value} {str(self.timedelta)} milliseconds",
                f"{APICallAttributes.FINISHED_AT.value} {str(self.timedelta + self.response_time)} milliseconds",
            ]
        random.shuffle(edge_attributes)
        return ",".join(edge_attributes)

    def to_edge_str_full_info(self):
        return "|".join(
            [self.rpc_id, self.rpc_type, self.src_ms, self.dest_ms, "ARRIVED_" + str(self.timedelta), "RT_" + str(self.response_time)]
        )

    @classmethod
    def get_api_call_from_trace(cls, edge_trace: str) -> APICall:
        pattern = re.compile("MS_(\d+)")
        edge_trace = re.sub(pattern, ms_repl_fn, edge_trace)
        columns = edge_trace.split("|")
        if len(columns) != 6:
            return APICall(
                rpc_id="",
                timedelta=-1,
                rpc_type="",
                src_ms="",
                dest_ms="",
                response_time=-1,
                invalid_reason=["num columns"],
            )

        invalid_reason = []
        rpc_id = columns[0]
        try:
            timedelta = int(parse("ARRIVED_{t}", columns[4])["t"])
        except TypeError:
            timedelta = -1
            invalid_reason.append("timedelta type error")
        rpc_type = columns[1]
        src = columns[2]
        dest = columns[3]
        try:
            response_time = int(parse("RT_{t}", columns[5])["t"])
        except TypeError:
            response_time = -1
            invalid_reason.append("response_time type error")

        return APICall(
            rpc_id=rpc_id,
            timedelta=timedelta,
            rpc_type=rpc_type,
            src_ms=src,
            dest_ms=dest,
            response_time=response_time,
            invalid_reason=invalid_reason,
        )
