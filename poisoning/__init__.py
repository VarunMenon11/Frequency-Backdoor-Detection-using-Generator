"""Poisoning pipelines and trigger construction."""

from poisoning.advanced_trigger import (
    AdvancedTriggerConfig,
    apply_advanced_trigger,
    build_frequency_band_mask,
    build_normalized_frequency_radius,
)
from poisoning.frequency_trigger import (
    FrequencyTriggerConfig,
    apply_frequency_trigger,
    build_frequency_pattern,
)
from poisoning.benchmark_trigger import (
    BenchmarkTriggerConfig,
    apply_benchmark_trigger,
    build_blended_carrier,
)

__all__ = [
    "AdvancedTriggerConfig",
    "FrequencyTriggerConfig",
    "apply_advanced_trigger",
    "apply_frequency_trigger",
    "build_frequency_band_mask",
    "build_frequency_pattern",
    "build_normalized_frequency_radius",
    "BenchmarkTriggerConfig",
    "apply_benchmark_trigger",
    "build_blended_carrier",
]
