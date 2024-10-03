from __future__ import annotations

from datetime import datetime, timedelta
from typing import List

from pydantic import BaseModel

from trace_gen.schema.call_graph import CallGraph


class Bucket(BaseModel):
    start: datetime
    duration: int  # ms
    num_requests: int
    call_graphs: List[CallGraph]

    def add_call_graph(self, call_graph: CallGraph) -> None:
        self.num_requests += 1
        self.call_graphs.append(call_graph)

    @classmethod
    def get_buckets_from_call_graphs(
        cls, call_graphs: List[CallGraph], bucket_window_ms: int = 10
    ) -> List[Bucket]:
        buckets = []

        bucket_start = call_graphs[0].arrived_at
        bucket_end = max([call_graph.arrived_at for call_graph in call_graphs])
        bucket_size = timedelta(microseconds=bucket_window_ms * 1000)

        search_idx = 0
        while bucket_start <= bucket_end:
            next_bucket_start = bucket_start + bucket_size
            curr_bucket = Bucket(
                start=bucket_start,
                duration=bucket_window_ms,
                num_requests=0,
                call_graphs=[],
            )
            for i in range(search_idx, len(call_graphs)):
                if (
                    call_graphs[i].arrived_at >= bucket_start
                    and call_graphs[i].arrived_at < next_bucket_start
                ):
                    curr_bucket.add_call_graph(call_graphs[i])
                    search_idx = i
            if len(curr_bucket.call_graphs) > 0:
                buckets.append(curr_bucket)
            bucket_start = next_bucket_start

        return buckets

    @classmethod
    def get_buckets_from_file(cls, fname: str) -> List[Bucket]:
        call_graphs = CallGraph.get_call_graphs_from_file(fname)
        return Bucket.get_buckets_from_call_graphs(
            call_graphs=call_graphs[:-1], bucket_window_ms=1
        )
