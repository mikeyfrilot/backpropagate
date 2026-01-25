"""
Backpropagation Engine - Core learning engine for tool feedback.

This module implements the main engine that processes traces, generates
learning signals, and updates memory based on feedback.

P0: Handle empty/malformed traces gracefully.
P2: Dry-run/analysis mode (engine.simulate).
P2: Explanation interface for learning decisions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable
import logging

from .contracts import LearningSignal, Trace, TraceStep, SignalSource
from .feedback import (
    score_feedback,
    score_binary_feedback,
    aggregate_signals,
)
from .trace import TraceGraph, TraceGraphBuilder
from .memory import MemoryUpdater, MemoryStats


logger = logging.getLogger(__name__)


@dataclass
class PropagationResult:
    """Result of a backpropagation operation."""

    signals_generated: int
    signals_applied: int
    tools_affected: list[str]
    trace_id: str
    success: bool
    errors: list[str] = field(default_factory=list)
    explanations: list[str] = field(default_factory=list)


@dataclass
class SimulationResult:
    """Result of a dry-run simulation."""

    would_generate: list[LearningSignal]
    would_affect: list[str]
    trace_id: str
    explanations: list[str]


class BackpropagationEngine:
    """
    Core engine for processing feedback and updating learning memory.

    The engine processes execution traces and feedback to generate
    learning signals that are then stored in memory.

    Features:
    - P0: Handles empty/malformed traces
    - P2: Dry-run/simulation mode
    - P2: Explanation interface
    """

    def __init__(
        self,
        memory: MemoryUpdater | None = None,
        propagation_factor: float = 0.5,
        min_confidence: float = 0.1,
    ) -> None:
        """
        Initialize the backpropagation engine.

        Args:
            memory: Memory updater instance (creates new if None)
            propagation_factor: How much signal propagates to earlier steps
            min_confidence: Minimum confidence to apply a signal
        """
        self.memory = memory or MemoryUpdater()
        self.propagation_factor = propagation_factor
        self.min_confidence = min_confidence
        self._signal_hooks: list[Callable[[LearningSignal], None]] = []

    def add_signal_hook(self, hook: Callable[[LearningSignal], None]) -> None:
        """Add a hook that's called when signals are generated."""
        self._signal_hooks.append(hook)

    def _validate_trace(self, trace: Trace) -> list[str]:
        """
        Validate a trace and return any errors.

        P0: Trace validation - require steps and success keys.
        """
        errors: list[str] = []

        if not trace.trace_id:
            errors.append("Trace missing trace_id")

        if not trace.steps:
            errors.append("Trace has no steps")

        for i, step in enumerate(trace.steps):
            if not step.tool_id:
                errors.append(f"Step {i} missing tool_id")

        return errors

    def _generate_signals_from_trace(
        self,
        trace: Trace,
        feedback_delta: float,
        source: SignalSource = SignalSource.TOOL_RESULT,
    ) -> tuple[list[LearningSignal], list[str]]:
        """
        Generate learning signals from a trace.

        Signals propagate backwards through the trace with decreasing
        confidence based on distance from the feedback point.

        Returns:
            Tuple of (signals, explanations)
        """
        signals: list[LearningSignal] = []
        explanations: list[str] = []

        if not trace.steps:
            return signals, explanations

        # Sort steps by order (latest first for backprop)
        sorted_steps = sorted(trace.steps, key=lambda s: s.order, reverse=True)

        for i, step in enumerate(sorted_steps):
            # Calculate propagated confidence
            # Latest step gets full confidence, earlier steps get reduced
            propagation_decay = self.propagation_factor ** i
            confidence = propagation_decay

            # Skip if below minimum confidence
            if confidence < self.min_confidence:
                explanations.append(
                    f"Skipped {step.tool_id}: confidence {confidence:.2f} < min {self.min_confidence}"
                )
                continue

            # Adjust delta based on step success
            step_delta = feedback_delta
            if not step.success:
                # Failed steps get negative signal regardless
                step_delta = min(step_delta, -0.5)
                explanations.append(
                    f"Adjusted {step.tool_id}: failed step, delta clamped to {step_delta:.2f}"
                )

            signal = LearningSignal(
                tool_id=step.tool_id,
                delta=step_delta,
                confidence=confidence,
                source=source,
                context={"trace_id": trace.trace_id, "step_order": step.order},
            )
            signals.append(signal)

            explanations.append(
                f"Generated signal for {step.tool_id}: "
                f"delta={step_delta:.2f}, confidence={confidence:.2f}"
            )

        return signals, explanations

    def propagate(
        self,
        trace: Trace,
        feedback_delta: float,
        source: SignalSource = SignalSource.USER_FEEDBACK,
        apply_to_memory: bool = True,
    ) -> PropagationResult:
        """
        Propagate feedback through a trace and update memory.

        Args:
            trace: The execution trace
            feedback_delta: The feedback delta to propagate
            source: Source of the feedback
            apply_to_memory: Whether to apply signals to memory

        Returns:
            PropagationResult with details of the operation
        """
        # P0: Validate trace
        errors = self._validate_trace(trace)
        if errors:
            logger.warning(f"Invalid trace {trace.trace_id}: {errors}")
            return PropagationResult(
                signals_generated=0,
                signals_applied=0,
                tools_affected=[],
                trace_id=trace.trace_id or "unknown",
                success=False,
                errors=errors,
            )

        # Generate signals
        signals, explanations = self._generate_signals_from_trace(
            trace, feedback_delta, source
        )

        tools_affected = list(set(s.tool_id for s in signals))

        # Apply to memory if requested
        applied_count = 0
        if apply_to_memory:
            for signal in signals:
                self.memory.update(signal)
                applied_count += 1

                # Call hooks
                for hook in self._signal_hooks:
                    try:
                        hook(signal)
                    except Exception as e:
                        logger.warning(f"Signal hook error: {e}")

        return PropagationResult(
            signals_generated=len(signals),
            signals_applied=applied_count,
            tools_affected=tools_affected,
            trace_id=trace.trace_id,
            success=True,
            explanations=explanations,
        )

    def propagate_from_graph(
        self,
        graph: TraceGraph,
        feedback_delta: float,
        source: SignalSource = SignalSource.USER_FEEDBACK,
    ) -> PropagationResult:
        """
        Propagate feedback through a trace graph.

        Args:
            graph: The trace graph
            feedback_delta: The feedback delta to propagate
            source: Source of the feedback

        Returns:
            PropagationResult with details
        """
        return self.propagate(graph.trace, feedback_delta, source)

    def simulate(
        self,
        trace: Trace,
        feedback_delta: float,
        source: SignalSource = SignalSource.USER_FEEDBACK,
    ) -> SimulationResult:
        """
        Simulate propagation without applying to memory.

        P2: Dry-run/analysis mode.

        Args:
            trace: The execution trace
            feedback_delta: The feedback delta to simulate
            source: Source of the feedback

        Returns:
            SimulationResult with what would happen
        """
        errors = self._validate_trace(trace)
        if errors:
            return SimulationResult(
                would_generate=[],
                would_affect=[],
                trace_id=trace.trace_id or "unknown",
                explanations=[f"Validation errors: {errors}"],
            )

        signals, explanations = self._generate_signals_from_trace(
            trace, feedback_delta, source
        )

        return SimulationResult(
            would_generate=signals,
            would_affect=list(set(s.tool_id for s in signals)),
            trace_id=trace.trace_id,
            explanations=explanations,
        )

    def process_feedback(
        self,
        trace: Trace,
        feedback_type: str,
        source: SignalSource = SignalSource.USER_FEEDBACK,
    ) -> PropagationResult:
        """
        Process string feedback (e.g., "thumbs_up") for a trace.

        Args:
            trace: The execution trace
            feedback_type: Type of feedback (e.g., "thumbs_up", "wrong")
            source: Source of the feedback

        Returns:
            PropagationResult with details
        """
        # Convert feedback type to a signal to get the delta
        # Use a dummy tool_id - we just need the delta
        signal = score_feedback(feedback_type, "dummy", source=source)
        return self.propagate(trace, signal.delta, source)

    def process_binary_feedback(
        self,
        trace: Trace,
        is_positive: bool,
        intensity: float = 0.5,
    ) -> PropagationResult:
        """
        Process binary (yes/no) feedback for a trace.

        Args:
            trace: The execution trace
            is_positive: Whether feedback is positive
            intensity: Strength of the feedback

        Returns:
            PropagationResult with details
        """
        signal = score_binary_feedback(is_positive, "dummy", intensity)
        return self.propagate(trace, signal.delta)

    def learn_from_success(self, trace: Trace) -> PropagationResult:
        """
        Apply positive learning from a successful trace.

        Args:
            trace: The successful execution trace

        Returns:
            PropagationResult with details
        """
        return self.propagate(trace, 0.5, SignalSource.TOOL_RESULT)

    def learn_from_failure(self, trace: Trace) -> PropagationResult:
        """
        Apply negative learning from a failed trace.

        Args:
            trace: The failed execution trace

        Returns:
            PropagationResult with details
        """
        return self.propagate(trace, -0.5, SignalSource.TOOL_RESULT)

    def get_tool_scores(self) -> dict[str, float]:
        """Get current scores for all tools."""
        return self.memory.get_all_scores()

    def get_tool_score(self, tool_id: str) -> float:
        """Get current score for a specific tool."""
        return self.memory.get_tool_score(tool_id)

    def explain_tool_score(self, tool_id: str) -> dict[str, Any]:
        """
        Explain how a tool's score was calculated.

        P2: Explanation interface.

        Args:
            tool_id: The tool to explain

        Returns:
            Dictionary with explanation details
        """
        signals = self.memory.get_signals(tool_id=tool_id)
        score = self.memory.get_tool_score(tool_id)

        if not signals:
            return {
                "tool_id": tool_id,
                "current_score": 0.0,
                "signal_count": 0,
                "explanation": "No learning signals recorded for this tool.",
            }

        positive_signals = [s for s in signals if s.delta > 0]
        negative_signals = [s for s in signals if s.delta < 0]
        neutral_signals = [s for s in signals if s.delta == 0]

        return {
            "tool_id": tool_id,
            "current_score": score,
            "signal_count": len(signals),
            "positive_signals": len(positive_signals),
            "negative_signals": len(negative_signals),
            "neutral_signals": len(neutral_signals),
            "average_confidence": sum(s.confidence for s in signals) / len(signals),
            "sources": list(set(s.source.value for s in signals)),
            "explanation": (
                f"Score {score:.2f} computed from {len(signals)} signals: "
                f"{len(positive_signals)} positive, {len(negative_signals)} negative. "
                f"Signals are weighted by confidence and time decay."
            ),
        }

    def get_stats(self) -> MemoryStats:
        """Get memory statistics."""
        return self.memory.get_stats()

    def reset(self, tool_id: str | None = None) -> int:
        """
        Reset learning for a tool or all tools.

        Args:
            tool_id: Optional specific tool to reset

        Returns:
            Number of signals cleared
        """
        return self.memory.clear(tool_id)

    def rollback_to(self, timestamp: datetime, tool_id: str | None = None) -> int:
        """
        Rollback learning to a specific time.

        Args:
            timestamp: Time to rollback to
            tool_id: Optional specific tool to rollback

        Returns:
            Number of signals removed
        """
        return self.memory.rollback(timestamp, tool_id)

    def save(self, path: str) -> None:
        """Save engine state to file."""
        self.memory.save(path)

    @classmethod
    def load(cls, path: str, **kwargs: Any) -> "BackpropagationEngine":
        """Load engine state from file."""
        memory = MemoryUpdater.load(path)
        return cls(memory=memory, **kwargs)


def create_engine(
    max_signals_per_tool: int = 1000,
    propagation_factor: float = 0.5,
    min_confidence: float = 0.1,
) -> BackpropagationEngine:
    """
    Create a new backpropagation engine with specified settings.

    Args:
        max_signals_per_tool: Maximum signals per tool in memory
        propagation_factor: How much signal propagates to earlier steps
        min_confidence: Minimum confidence to apply a signal

    Returns:
        Configured BackpropagationEngine
    """
    memory = MemoryUpdater(max_signals_per_tool=max_signals_per_tool)
    return BackpropagationEngine(
        memory=memory,
        propagation_factor=propagation_factor,
        min_confidence=min_confidence,
    )
