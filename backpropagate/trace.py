"""
Trace Graph - Build and analyze tool execution traces.

This module handles the construction and serialization of trace graphs
representing tool execution chains for learning analysis.

P1: Explicit ordering to trace graph nodes.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterator
import json

from .contracts import Trace, TraceStep, SCHEMA_VERSION


@dataclass
class TraceNode:
    """
    A node in the trace graph representing a tool execution.

    Attributes:
        step: The trace step this node represents
        children: Child nodes (tools called after this one)
        parent: Parent node (tool that called before this one)
    """

    step: TraceStep
    children: list["TraceNode"] = field(default_factory=list)
    parent: "TraceNode | None" = None

    @property
    def tool_id(self) -> str:
        """Get the tool ID for this node."""
        return self.step.tool_id

    @property
    def success(self) -> bool:
        """Get success status for this node."""
        return self.step.success

    @property
    def order(self) -> int:
        """Get execution order for this node."""
        return self.step.order

    def add_child(self, node: "TraceNode") -> None:
        """Add a child node."""
        node.parent = self
        self.children.append(node)

    def depth(self) -> int:
        """Calculate depth from root."""
        depth = 0
        current = self.parent
        while current is not None:
            depth += 1
            current = current.parent
        return depth


class TraceGraph:
    """
    A graph representation of tool execution traces.

    Supports building trace graphs from execution data and
    serialization for storage and analysis.

    Attributes:
        trace: The underlying trace data
        nodes: All nodes in the graph indexed by order
        root: The root node of the trace graph
    """

    def __init__(self, trace: Trace) -> None:
        """
        Initialize a trace graph from a trace.

        Args:
            trace: The trace to build a graph from
        """
        self.trace = trace
        self.nodes: dict[int, TraceNode] = {}
        self.root: TraceNode | None = None
        self._build_graph()

    def _build_graph(self) -> None:
        """Build the graph structure from trace steps."""
        if not self.trace.steps:
            return

        # Sort steps by order (P1: explicit ordering)
        sorted_steps = sorted(self.trace.steps, key=lambda s: s.order)

        # Create nodes for each step
        for step in sorted_steps:
            node = TraceNode(step=step)
            self.nodes[step.order] = node

        # Build parent-child relationships (sequential by default)
        for i, step in enumerate(sorted_steps):
            node = self.nodes[step.order]
            if i == 0:
                self.root = node
            else:
                prev_order = sorted_steps[i - 1].order
                parent = self.nodes[prev_order]
                parent.add_child(node)

    @property
    def trace_id(self) -> str:
        """Get the trace ID."""
        return self.trace.trace_id

    @property
    def success(self) -> bool:
        """Get overall trace success status."""
        return self.trace.success

    def get_node(self, order: int) -> TraceNode | None:
        """Get a node by its order."""
        return self.nodes.get(order)

    def get_tool_nodes(self, tool_id: str) -> list[TraceNode]:
        """Get all nodes for a specific tool."""
        return [n for n in self.nodes.values() if n.tool_id == tool_id]

    def get_failed_nodes(self) -> list[TraceNode]:
        """Get all nodes that failed."""
        return [n for n in self.nodes.values() if not n.success]

    def get_successful_nodes(self) -> list[TraceNode]:
        """Get all nodes that succeeded."""
        return [n for n in self.nodes.values() if n.success]

    def iter_nodes(self) -> Iterator[TraceNode]:
        """Iterate over nodes in execution order."""
        for order in sorted(self.nodes.keys()):
            yield self.nodes[order]

    def iter_depth_first(self) -> Iterator[TraceNode]:
        """Iterate over nodes in depth-first order."""
        if self.root is None:
            return

        stack = [self.root]
        while stack:
            node = stack.pop()
            yield node
            # Add children in reverse order for correct DFS
            stack.extend(reversed(node.children))

    def iter_breadth_first(self) -> Iterator[TraceNode]:
        """Iterate over nodes in breadth-first order."""
        if self.root is None:
            return

        from collections import deque

        queue = deque([self.root])
        while queue:
            node = queue.popleft()
            yield node
            queue.extend(node.children)

    def get_execution_path(self) -> list[str]:
        """Get the tool execution path as a list of tool IDs."""
        return [node.tool_id for node in self.iter_nodes()]

    def get_failure_path(self) -> list[str]:
        """Get path leading to first failure, if any."""
        for node in self.iter_nodes():
            if not node.success:
                # Build path from root to this node
                path = []
                current: TraceNode | None = node
                while current is not None:
                    path.insert(0, current.tool_id)
                    current = current.parent
                return path
        return []

    def to_dict(self) -> dict[str, Any]:
        """Serialize the graph to a dictionary."""
        return {
            "trace": self.trace.to_dict(),
            "node_count": len(self.nodes),
            "execution_path": self.get_execution_path(),
            "has_failures": len(self.get_failed_nodes()) > 0,
        }

    def to_json(self, indent: int | None = 2) -> str:
        """Serialize the graph to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TraceGraph":
        """Deserialize from a dictionary."""
        trace = Trace.from_dict(data["trace"])
        return cls(trace)

    @classmethod
    def from_json(cls, json_str: str) -> "TraceGraph":
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def __len__(self) -> int:
        """Return number of nodes in the graph."""
        return len(self.nodes)

    def __bool__(self) -> bool:
        """Return True if graph has nodes."""
        return len(self.nodes) > 0


class TraceGraphBuilder:
    """
    Builder for constructing trace graphs incrementally.

    Use this to build traces step-by-step during execution.
    """

    def __init__(self, trace_id: str, metadata: dict[str, Any] | None = None) -> None:
        """
        Initialize a new trace graph builder.

        Args:
            trace_id: Unique identifier for this trace
            metadata: Optional trace metadata
        """
        self.trace_id = trace_id
        self.metadata = metadata or {}
        self.steps: list[TraceStep] = []
        self.success = True
        self.created_at = datetime.utcnow()

    def add_step(
        self,
        tool_id: str,
        input_data: dict[str, Any],
        output_data: dict[str, Any],
        success: bool,
        duration_ms: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> TraceStep:
        """
        Add a step to the trace.

        Args:
            tool_id: Identifier for the tool
            input_data: Input provided to the tool
            output_data: Output from the tool
            success: Whether the step succeeded
            duration_ms: Execution time in milliseconds
            metadata: Additional step metadata

        Returns:
            The created TraceStep
        """
        step = TraceStep(
            tool_id=tool_id,
            input_data=input_data,
            output_data=output_data,
            success=success,
            duration_ms=duration_ms,
            order=len(self.steps),
            metadata=metadata or {},
        )
        self.steps.append(step)

        # Update overall success
        if not success:
            self.success = False

        return step

    def build_trace(self) -> Trace:
        """Build the final Trace object."""
        return Trace(
            trace_id=self.trace_id,
            steps=self.steps.copy(),
            success=self.success,
            created_at=self.created_at,
            metadata=self.metadata,
        )

    def build(self) -> TraceGraph:
        """Build the final TraceGraph."""
        trace = self.build_trace()
        return TraceGraph(trace)

    def __len__(self) -> int:
        """Return number of steps added."""
        return len(self.steps)


def build_trace_from_steps(
    trace_id: str,
    steps: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> TraceGraph:
    """
    Build a trace graph from a list of step dictionaries.

    Args:
        trace_id: Unique identifier for the trace
        steps: List of step data dictionaries
        metadata: Optional trace metadata

    Returns:
        Constructed TraceGraph
    """
    builder = TraceGraphBuilder(trace_id, metadata)

    for step_data in steps:
        builder.add_step(
            tool_id=step_data["tool_id"],
            input_data=step_data.get("input_data", {}),
            output_data=step_data.get("output_data", {}),
            success=step_data.get("success", True),
            duration_ms=step_data.get("duration_ms", 0.0),
            metadata=step_data.get("metadata", {}),
        )

    return builder.build()


def get_tool_statistics(graphs: list[TraceGraph]) -> dict[str, dict[str, Any]]:
    """
    Calculate statistics for tools across multiple traces.

    Args:
        graphs: List of trace graphs to analyze

    Returns:
        Dictionary mapping tool_id to statistics
    """
    stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_duration_ms": 0.0,
            "success_rate": 0.0,
        }
    )

    for graph in graphs:
        for node in graph.iter_nodes():
            tool_stats = stats[node.tool_id]
            tool_stats["total_calls"] += 1
            tool_stats["total_duration_ms"] += node.step.duration_ms
            if node.success:
                tool_stats["successful_calls"] += 1
            else:
                tool_stats["failed_calls"] += 1

    # Calculate success rates
    for tool_id, tool_stats in stats.items():
        if tool_stats["total_calls"] > 0:
            tool_stats["success_rate"] = (
                tool_stats["successful_calls"] / tool_stats["total_calls"]
            )

    return dict(stats)
