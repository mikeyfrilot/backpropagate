"""
Feedback Scoring - Convert user feedback to learning signals.

This module handles the conversion of various feedback types
into normalized learning signals for the backpropagation system.

P1: Confidence weighting based on feedback source and recency.
"""

from datetime import datetime, timedelta
from typing import Any

from .contracts import LearningSignal, SignalSource, MIN_DELTA, MAX_DELTA


# Feedback type to delta mappings
FEEDBACK_DELTAS = {
    # Positive feedback
    "thumbs_up": 0.5,
    "like": 0.5,
    "love": 0.8,
    "helpful": 0.6,
    "correct": 0.7,
    "perfect": 1.0,
    "good": 0.4,
    "yes": 0.3,
    # Negative feedback
    "thumbs_down": -0.5,
    "dislike": -0.5,
    "wrong": -0.7,
    "unhelpful": -0.6,
    "incorrect": -0.8,
    "bad": -0.4,
    "no": -0.3,
    "error": -0.6,
    # Neutral
    "neutral": 0.0,
    "skip": 0.0,
    "unknown": 0.0,
}

# Confidence weights by source
SOURCE_CONFIDENCE = {
    SignalSource.USER_FEEDBACK: 1.0,
    SignalSource.CORRECTION: 0.9,
    SignalSource.TOOL_RESULT: 0.7,
    SignalSource.INFERENCE: 0.5,
    SignalSource.SYSTEM: 0.8,
}

# Decay half-life in hours for confidence over time
CONFIDENCE_DECAY_HALFLIFE_HOURS = 168  # 1 week


def score_feedback(
    feedback_type: str,
    tool_id: str,
    source: SignalSource = SignalSource.USER_FEEDBACK,
    context: dict[str, Any] | None = None,
    input_pattern: str | None = None,
    timestamp: datetime | None = None,
    custom_delta: float | None = None,
) -> LearningSignal:
    """
    Convert feedback into a learning signal.

    Args:
        feedback_type: Type of feedback (e.g., "thumbs_up", "wrong")
        tool_id: Identifier for the tool receiving feedback
        source: Source of the feedback signal
        context: Optional context data
        input_pattern: Optional pattern for scoped learning
        timestamp: When the feedback was given (defaults to now)
        custom_delta: Override the default delta for this feedback type

    Returns:
        LearningSignal with appropriate delta and confidence

    Examples:
        >>> signal = score_feedback("thumbs_up", "search_tool")
        >>> signal.delta
        0.5
        >>> signal = score_feedback("wrong", "query_tool")
        >>> signal.delta
        -0.7
    """
    # Get delta from mappings or use custom
    if custom_delta is not None:
        delta = max(MIN_DELTA, min(MAX_DELTA, custom_delta))
    else:
        delta = FEEDBACK_DELTAS.get(feedback_type.lower(), 0.0)

    # Calculate confidence based on source
    confidence = SOURCE_CONFIDENCE.get(source, 0.5)

    return LearningSignal(
        tool_id=tool_id,
        delta=delta,
        confidence=confidence,
        source=source,
        timestamp=timestamp or datetime.utcnow(),
        context=context or {},
        input_pattern=input_pattern,
    )


def score_binary_feedback(
    is_positive: bool,
    tool_id: str,
    intensity: float = 0.5,
    source: SignalSource = SignalSource.USER_FEEDBACK,
    context: dict[str, Any] | None = None,
) -> LearningSignal:
    """
    Convert binary (yes/no) feedback to a learning signal.

    Args:
        is_positive: Whether the feedback is positive
        tool_id: Identifier for the tool
        intensity: Strength of the feedback (0.0 to 1.0)
        source: Source of the feedback
        context: Optional context data

    Returns:
        LearningSignal with calculated delta
    """
    intensity = max(0.0, min(1.0, intensity))
    delta = intensity if is_positive else -intensity

    return LearningSignal(
        tool_id=tool_id,
        delta=delta,
        confidence=SOURCE_CONFIDENCE.get(source, 0.5),
        source=source,
        context=context or {},
    )


def score_numeric_feedback(
    rating: float,
    tool_id: str,
    min_rating: float = 1.0,
    max_rating: float = 5.0,
    source: SignalSource = SignalSource.USER_FEEDBACK,
    context: dict[str, Any] | None = None,
) -> LearningSignal:
    """
    Convert a numeric rating to a learning signal.

    Args:
        rating: The numeric rating value
        tool_id: Identifier for the tool
        min_rating: Minimum possible rating
        max_rating: Maximum possible rating
        source: Source of the feedback
        context: Optional context data

    Returns:
        LearningSignal with delta normalized to [-1.0, +1.0]

    Example:
        >>> signal = score_numeric_feedback(5, "tool", min_rating=1, max_rating=5)
        >>> signal.delta
        1.0
        >>> signal = score_numeric_feedback(3, "tool", min_rating=1, max_rating=5)
        >>> signal.delta
        0.0
    """
    # Normalize rating to [0, 1] then scale to [-1, 1]
    normalized = (rating - min_rating) / (max_rating - min_rating)
    normalized = max(0.0, min(1.0, normalized))
    delta = (normalized * 2) - 1  # Scale to [-1, 1]

    return LearningSignal(
        tool_id=tool_id,
        delta=delta,
        confidence=SOURCE_CONFIDENCE.get(source, 0.5),
        source=source,
        context=context or {},
    )


def apply_time_decay(
    signal: LearningSignal,
    reference_time: datetime | None = None,
    halflife_hours: float = CONFIDENCE_DECAY_HALFLIFE_HOURS,
) -> LearningSignal:
    """
    Apply time-based confidence decay to a signal.

    Older signals have reduced confidence using exponential decay.

    Args:
        signal: The learning signal to decay
        reference_time: Time to measure decay from (defaults to now)
        halflife_hours: Hours for confidence to decay by half

    Returns:
        New LearningSignal with decayed confidence
    """
    reference = reference_time or datetime.utcnow()
    age_hours = (reference - signal.timestamp).total_seconds() / 3600

    # Exponential decay: confidence * (0.5 ^ (age / halflife))
    decay_factor = 0.5 ** (age_hours / halflife_hours)
    decayed_confidence = signal.confidence * decay_factor

    return LearningSignal(
        tool_id=signal.tool_id,
        delta=signal.delta,
        confidence=decayed_confidence,
        source=signal.source,
        timestamp=signal.timestamp,
        context=signal.context,
        input_pattern=signal.input_pattern,
        schema_version=signal.schema_version,
    )


def aggregate_signals(
    signals: list[LearningSignal],
    apply_decay: bool = True,
) -> float:
    """
    Aggregate multiple signals into a single weighted delta.

    Args:
        signals: List of learning signals to aggregate
        apply_decay: Whether to apply time decay to signals

    Returns:
        Weighted average delta clamped to [-1.0, +1.0]
    """
    if not signals:
        return 0.0

    # Apply decay if requested
    if apply_decay:
        signals = [apply_time_decay(s) for s in signals]

    # Weighted average by confidence
    total_weight = sum(s.confidence for s in signals)
    if total_weight == 0:
        return 0.0

    weighted_sum = sum(s.weighted_delta() for s in signals)
    result = weighted_sum / total_weight

    return max(MIN_DELTA, min(MAX_DELTA, result))


def is_positive_feedback(feedback_type: str) -> bool:
    """Check if a feedback type is positive."""
    delta = FEEDBACK_DELTAS.get(feedback_type.lower(), 0.0)
    return delta > 0


def is_negative_feedback(feedback_type: str) -> bool:
    """Check if a feedback type is negative."""
    delta = FEEDBACK_DELTAS.get(feedback_type.lower(), 0.0)
    return delta < 0


def is_neutral_feedback(feedback_type: str) -> bool:
    """Check if a feedback type is neutral."""
    delta = FEEDBACK_DELTAS.get(feedback_type.lower(), 0.0)
    return delta == 0.0


def get_feedback_types() -> dict[str, float]:
    """Get all supported feedback types and their deltas."""
    return FEEDBACK_DELTAS.copy()
