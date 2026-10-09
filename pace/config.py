"""Configuration for the running pose demo."""

from dataclasses import dataclass


@dataclass
class AnalysisConfig:
    min_visibility: float = 0.45
    smoothing_window_seconds: float = 0.18
    min_event_interval_seconds: float = 0.22
    min_stride_seconds: float = 0.35
    max_stride_seconds: float = 2.0
    ground_percentile: float = 72.0
    strike_velocity_tolerance: float = 0.42
    event_display_seconds: float = 0.12
    cadence_low: float = 150.0
    cadence_high: float = 200.0
    asymmetry_warning_percent: float = 10.0
    overstride_warning_ratio: float = 0.18

