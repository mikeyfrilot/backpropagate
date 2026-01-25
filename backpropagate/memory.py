"""
Memory Updater - Manage and update learning memory.

This module handles storing, retrieving, and updating learning signals
in memory with support for rollback and compaction.

P0: Make learning reversible (store raw signals, compute dynamically).
P1: Tool isolation (scope learning per tool/input pattern).
P1: Max-memory cap (rolling window/compaction).
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Iterator
import json
from pathlib import Path

from .contracts import LearningSignal, SCHEMA_VERSION
from .feedback import apply_time_decay, aggregate_signals


# Default memory limits
DEFAULT_MAX_SIGNALS_PER_TOOL = 1000
DEFAULT_COMPACTION_THRESHOLD = 0.8  # Compact when 80% full
DEFAULT_ROLLING_WINDOW_DAYS = 30


@dataclass
class MemoryStats:
    """Statistics about memory usage."""

    total_signals: int
    total_tools: int
    signals_by_tool: dict[str, int]
    oldest_signal: datetime | None
    newest_signal: datetime | None
    memory_utilization: float  # 0.0 to 1.0


@dataclass
class ToolMemory:
    """
    Memory storage for a single tool.

    Stores raw signals and computes scores dynamically.
    Supports rollback by keeping signal history.
    """

    tool_id: str
    signals: list[LearningSignal] = field(default_factory=list)
    max_signals: int = DEFAULT_MAX_SIGNALS_PER_TOOL
    created_at: datetime = field(default_factory=datetime.utcnow)

    def add_signal(self, signal: LearningSignal) -> None:
        """Add a signal, triggering compaction if needed."""
        self.signals.append(signal)
        if len(self.signals) > self.max_signals:
            self._compact()

    def _compact(self) -> None:
        """Compact signals by keeping recent ones."""
        # Keep the newest half of signals
        keep_count = self.max_signals // 2
        self.signals = sorted(
            self.signals, key=lambda s: s.timestamp, reverse=True
        )[:keep_count]

    def get_current_score(self, apply_decay: bool = True) -> float:
        """
        Compute current score dynamically from raw signals.

        P0: Reversible learning - score is computed, not stored.
        """
        return aggregate_signals(self.signals, apply_decay=apply_decay)

    def get_signals_for_pattern(self, pattern: str) -> list[LearningSignal]:
        """Get signals matching an input pattern."""
        return [s for s in self.signals if s.input_pattern == pattern]

    def get_signals_since(self, since: datetime) -> list[LearningSignal]:
        """Get signals since a given time."""
        return [s for s in self.signals if s.timestamp >= since]

    def rollback_to(self, timestamp: datetime) -> int:
        """
        Rollback signals to a given timestamp.

        Returns the number of signals removed.
        """
        original_count = len(self.signals)
        self.signals = [s for s in self.signals if s.timestamp <= timestamp]
        return original_count - len(self.signals)

    def clear(self) -> int:
        """Clear all signals, returning count removed."""
        count = len(self.signals)
        self.signals = []
        return count

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "tool_id": self.tool_id,
            "signals": [s.to_dict() for s in self.signals],
            "max_signals": self.max_signals,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolMemory":
        """Deserialize from dictionary."""
        memory = cls(
            tool_id=data["tool_id"],
            max_signals=data.get("max_signals", DEFAULT_MAX_SIGNALS_PER_TOOL),
            created_at=datetime.fromisoformat(data["created_at"])
            if "created_at" in data
            else datetime.utcnow(),
        )
        memory.signals = [
            LearningSignal.from_dict(s) for s in data.get("signals", [])
        ]
        return memory


class MemoryUpdater:
    """
    Manages learning memory for multiple tools.

    Features:
    - P0: Reversible learning via raw signal storage
    - P1: Tool isolation with per-tool memory
    - P1: Max-memory cap with rolling window
    """

    def __init__(
        self,
        max_signals_per_tool: int = DEFAULT_MAX_SIGNALS_PER_TOOL,
        rolling_window_days: int = DEFAULT_ROLLING_WINDOW_DAYS,
    ) -> None:
        """
        Initialize memory updater.

        Args:
            max_signals_per_tool: Maximum signals to keep per tool
            rolling_window_days: Days to keep signals before expiry
        """
        self.max_signals_per_tool = max_signals_per_tool
        self.rolling_window_days = rolling_window_days
        self._tool_memories: dict[str, ToolMemory] = {}
        self._schema_version = SCHEMA_VERSION

    def update(self, signal: LearningSignal) -> None:
        """
        Update memory with a new learning signal.

        Args:
            signal: The learning signal to add
        """
        tool_id = signal.tool_id

        # Create tool memory if needed (P1: tool isolation)
        if tool_id not in self._tool_memories:
            self._tool_memories[tool_id] = ToolMemory(
                tool_id=tool_id,
                max_signals=self.max_signals_per_tool,
            )

        self._tool_memories[tool_id].add_signal(signal)

    def update_batch(self, signals: list[LearningSignal]) -> int:
        """
        Update memory with multiple signals.

        Args:
            signals: List of learning signals to add

        Returns:
            Number of signals added
        """
        for signal in signals:
            self.update(signal)
        return len(signals)

    def get_tool_score(
        self,
        tool_id: str,
        apply_decay: bool = True,
        pattern: str | None = None,
    ) -> float:
        """
        Get current learning score for a tool.

        Args:
            tool_id: The tool to get score for
            apply_decay: Whether to apply time decay
            pattern: Optional pattern to filter signals

        Returns:
            Aggregated score in range [-1.0, +1.0]
        """
        if tool_id not in self._tool_memories:
            return 0.0

        memory = self._tool_memories[tool_id]

        if pattern:
            signals = memory.get_signals_for_pattern(pattern)
            return aggregate_signals(signals, apply_decay=apply_decay)

        return memory.get_current_score(apply_decay=apply_decay)

    def get_all_scores(self, apply_decay: bool = True) -> dict[str, float]:
        """
        Get scores for all tools.

        Returns:
            Dictionary mapping tool_id to score
        """
        return {
            tool_id: memory.get_current_score(apply_decay)
            for tool_id, memory in self._tool_memories.items()
        }

    def get_signals(
        self,
        tool_id: str | None = None,
        since: datetime | None = None,
        pattern: str | None = None,
    ) -> list[LearningSignal]:
        """
        Get signals with optional filtering.

        Args:
            tool_id: Filter by tool ID
            since: Filter by timestamp
            pattern: Filter by input pattern

        Returns:
            List of matching signals
        """
        signals: list[LearningSignal] = []

        memories = (
            [self._tool_memories[tool_id]]
            if tool_id and tool_id in self._tool_memories
            else self._tool_memories.values()
        )

        for memory in memories:
            tool_signals = memory.signals

            if since:
                tool_signals = [s for s in tool_signals if s.timestamp >= since]

            if pattern:
                tool_signals = [s for s in tool_signals if s.input_pattern == pattern]

            signals.extend(tool_signals)

        return signals

    def rollback(
        self,
        timestamp: datetime,
        tool_id: str | None = None,
    ) -> int:
        """
        Rollback signals to a given timestamp.

        P0: Reversible learning support.

        Args:
            timestamp: Rollback to this time
            tool_id: Optional specific tool to rollback

        Returns:
            Number of signals removed
        """
        total_removed = 0

        if tool_id:
            if tool_id in self._tool_memories:
                total_removed = self._tool_memories[tool_id].rollback_to(timestamp)
        else:
            for memory in self._tool_memories.values():
                total_removed += memory.rollback_to(timestamp)

        return total_removed

    def apply_rolling_window(self) -> int:
        """
        Remove signals older than the rolling window.

        P1: Max-memory cap via rolling window.

        Returns:
            Number of signals removed
        """
        cutoff = datetime.utcnow() - timedelta(days=self.rolling_window_days)
        return self.rollback(cutoff)

    def compact(self, tool_id: str | None = None) -> None:
        """
        Compact memory by removing old signals.

        Args:
            tool_id: Optional specific tool to compact
        """
        if tool_id:
            if tool_id in self._tool_memories:
                self._tool_memories[tool_id]._compact()
        else:
            for memory in self._tool_memories.values():
                memory._compact()

    def clear(self, tool_id: str | None = None) -> int:
        """
        Clear signals from memory.

        Args:
            tool_id: Optional specific tool to clear

        Returns:
            Number of signals removed
        """
        if tool_id:
            if tool_id in self._tool_memories:
                count = self._tool_memories[tool_id].clear()
                del self._tool_memories[tool_id]
                return count
            return 0
        else:
            total = sum(m.clear() for m in self._tool_memories.values())
            self._tool_memories.clear()
            return total

    def get_stats(self) -> MemoryStats:
        """Get memory statistics."""
        all_signals: list[LearningSignal] = []
        signals_by_tool: dict[str, int] = {}

        for tool_id, memory in self._tool_memories.items():
            signals_by_tool[tool_id] = len(memory.signals)
            all_signals.extend(memory.signals)

        timestamps = [s.timestamp for s in all_signals]

        total_capacity = len(self._tool_memories) * self.max_signals_per_tool
        utilization = len(all_signals) / total_capacity if total_capacity > 0 else 0.0

        return MemoryStats(
            total_signals=len(all_signals),
            total_tools=len(self._tool_memories),
            signals_by_tool=signals_by_tool,
            oldest_signal=min(timestamps) if timestamps else None,
            newest_signal=max(timestamps) if timestamps else None,
            memory_utilization=utilization,
        )

    def iter_tools(self) -> Iterator[str]:
        """Iterate over tool IDs in memory."""
        return iter(self._tool_memories.keys())

    def to_dict(self) -> dict[str, Any]:
        """Serialize memory to dictionary."""
        return {
            "schema_version": self._schema_version,
            "max_signals_per_tool": self.max_signals_per_tool,
            "rolling_window_days": self.rolling_window_days,
            "tool_memories": {
                tool_id: memory.to_dict()
                for tool_id, memory in self._tool_memories.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryUpdater":
        """Deserialize from dictionary."""
        updater = cls(
            max_signals_per_tool=data.get(
                "max_signals_per_tool", DEFAULT_MAX_SIGNALS_PER_TOOL
            ),
            rolling_window_days=data.get(
                "rolling_window_days", DEFAULT_ROLLING_WINDOW_DAYS
            ),
        )

        for tool_id, memory_data in data.get("tool_memories", {}).items():
            updater._tool_memories[tool_id] = ToolMemory.from_dict(memory_data)

        return updater

    def save(self, path: str | Path) -> None:
        """Save memory to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str | Path) -> "MemoryUpdater":
        """Load memory from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def __len__(self) -> int:
        """Return total number of signals."""
        return sum(len(m.signals) for m in self._tool_memories.values())

    def __contains__(self, tool_id: str) -> bool:
        """Check if tool has memory."""
        return tool_id in self._tool_memories
