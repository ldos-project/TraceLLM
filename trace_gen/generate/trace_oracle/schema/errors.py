from trace_gen.generate.trace_oracle.schema.task_type import GraphGenInstruction
from pydantic import BaseModel


class GenerationError(BaseModel):
    error_code: int
    detail: str

    def __str__(self):
        return f"Generation Error - {self.error_code}: {self.detail}"


class InvalidInstruction(GenerationError):
    @classmethod
    def from_args(cls, instruction: GraphGenInstruction):
        return InvalidInstruction(error_code=0, detail=f"Invalid Instruction - {instruction}")

class DeepCallGraph(GenerationError):
    @classmethod
    def from_args(cls, current_depth: int):
        return DeepCallGraph(error_code=1, detail=f"Subgraph instructions generated, but reached max depth: {current_depth}")


class ShallowCallGraph(GenerationError):
    @classmethod
    def from_args(cls, desired_depth: int, current_depth: int):
        return ShallowCallGraph(error_code=2, detail=f"No more subgraphs - desired depth: {desired_depth}, current_depth: {current_depth}")


class InvalidNumEdges(GenerationError):
    @classmethod
    def from_args(cls, desired_num_edges: int, current_num_edges: int):
        return InvalidNumEdges(error_code=3, detail=f"Invalid num edges - desired: {desired_num_edges}, current: {current_num_edges}")

class WrongEdgeStartRPCID(GenerationError):
    @classmethod
    def from_args(cls, desired: str, current: str):
        return WrongEdgeStartRPCID(error_code=4, detail=f"Invalid start RPC ID in generated edges - desired: {desired}, current: {current}")

class WrongSubgraphStartRPCID(GenerationError):
    @classmethod
    def from_args(cls, desired: list[str], generated: str):
        return WrongSubgraphStartRPCID(error_code=5, detail=f"Invalid start RPC ID in generated subgraph instructions - desired: {','.join(desired)}, generated: {generated}")


class WrongSubgraphStartNode(GenerationError):
    @classmethod
    def from_args(cls, desired: str, current: str):
        return WrongSubgraphStartNode(error_code=6, detail=f"Invalid start node in the subgraph instruction - desired: {desired}, current: {current}")


class LargeOffsetInSubgraphInst(GenerationError):
    @classmethod
    def from_args(cls, offset: int, latency: int):
        return LargeOffsetInSubgraphInst(error_code=7, detail=f"Offset is larger than latency in subgraph instruction - offset: {offset}, latency: {latency}")


class LargeLatencyInSubgraphs(GenerationError):
    @classmethod
    def from_args(cls, parent_latency: int, child_latency: int):
        return LargeLatencyInSubgraphs(error_code=8, detail=f"Parent latency > Child latency - parent: {parent_latency}, child: {child_latency}")


class WrongMaxDepthInSubgraphs(GenerationError):
    @classmethod
    def from_args(cls, parent_max_depth: int, child_max_depth: int):
        return WrongMaxDepthInSubgraphs(error_code=9, detail=f"Parent depth < Child depth - parent: {parent_max_depth}, child: {child_max_depth}")


class RemainingDepthLargerThanEdges(GenerationError):
    @classmethod
    def from_args(cls, remaining_depth: int, remaining_edges: int):
        return RemainingDepthLargerThanEdges(error_code=10, detail=f"Remaining depth > edges - depth: {remaining_depth}, edges: {remaining_edges}")


class NoSubgraphInst(GenerationError):
    @classmethod
    def from_args(cls, desired_depth: int, current_depth: int):
        return NoSubgraphInst(error_code=11, detail=f"No subgraph generation - desired depth: {desired_depth}, current_depth: {current_depth}")


class LargeLatencyInEdges(GenerationError):
    @classmethod
    def from_args(cls, desired_latency: int, generated_latency: int):
        return LargeLatencyInEdges(error_code=12, detail=f"Generated latency is larger than instruction's - desired: {desired_latency}, generated: {generated_latency}")


class MoreEdgesGenerated(GenerationError):
    @classmethod
    def from_args(cls, desired_edges: int, remaining_depth: int, num_generated_edges: int):
        return MoreEdgesGenerated(error_code=13, detail=f"More edges generated than required - desired edges: {desired_edges}, remaining_depth: {remaining_depth}, num_generated_edges: {num_generated_edges}")

class EdgesNotSplitted(GenerationError):
    @classmethod
    def from_args(cls, desired_edges: int, num_generated_edges: list[int]):
        return MoreEdgesGenerated(error_code=19, detail=f"Num edges not splitted - desired edges: {desired_edges}, num_generated_edges: {','.join(num_generated_edges)}")


class ParsingException(GenerationError):
    @classmethod
    def from_args(cls, exception_error: str):
        return ParsingException(error_code=14, detail=f"Error during parsing - {exception_error}")


class InvalidCallGraph(GenerationError):
    @classmethod
    def from_args(cls, reasons):
        return InvalidCallGraph(error_code=15, detail=f"Invalid Call Graph: {','.join(reasons)}")

class SubgraphWithLessDepth(GenerationError):
    @classmethod
    def from_args(cls, instruction_depth: int, subgraph_depths: list[int]):
        return SubgraphWithLessDepth(error_code=16, detail=f"All subgraphs generated shallow call graphs - instruction: {instruction_depth}, subgraphs: {','.join([str(d) for d in subgraph_depths])}")

class InvalidMaxDepth(GenerationError):
    @classmethod
    def from_args(cls, desired_depth: int, current_depth: int):
        return InvalidNumEdges(error_code=17, detail=f"Invalid max depth - desired: {desired_depth}, current: {current_depth}")

class SubgraphDepthNotSameWhileSplitting(GenerationError):
    @classmethod
    def from_args(cls, instruction_depth: int, subgraph_depths: list[int]):
        return SubgraphDepthNotSameWhileSplitting(error_code=18, detail=f"Depth should be the same when splitting - instruction: {instruction_depth}, subgraphs: {','.join([str(d) for d in subgraph_depths])}")

