"""Data models and schemas for the SRE Workflow Agent."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SeverityLevel(str, Enum):
    """Incident severity levels."""
    
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class LogLevel(str, Enum):
    """Log levels."""
    
    ERROR = "error"
    WARN = "warn"
    INFO = "info"
    DEBUG = "debug"
    NOTICE = "notice"


class IncidentStatus(str, Enum):
    """Incident status types."""
    
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"


class LogEntry(BaseModel):
    """Apache log entry model."""
    
    line_id: int = Field(..., description="Unique line identifier")
    timestamp: datetime = Field(..., description="Log entry timestamp")
    level: LogLevel = Field(..., description="Log level")
    content: str = Field(..., description="Log message content")
    event_id: str = Field(..., description="Event template ID")
    event_template: str = Field(..., description="Event template pattern")


class MetricData(BaseModel):
    """Metrics data model."""
    
    metric_name: str = Field(..., description="Name of the metric")
    value: float = Field(..., description="Metric value")
    timestamp: datetime = Field(..., description="Metric timestamp")
    labels: Dict[str, str] = Field(default_factory=dict, description="Metric labels")


class TraceData(BaseModel):
    """Trace data model."""
    
    trace_id: str = Field(..., description="Unique trace identifier")
    span_id: str = Field(..., description="Span identifier")
    operation_name: str = Field(..., description="Operation name")
    duration_ms: float = Field(..., description="Duration in milliseconds")
    status: str = Field(..., description="Trace status")
    timestamp: datetime = Field(..., description="Trace timestamp")


class IncidentRequest(BaseModel):
    """Incident analysis request model."""
    
    incident_id: str = Field(..., description="Unique incident identifier")
    title: str = Field(..., description="Incident title")
    description: str = Field(..., description="Incident description")
    severity: SeverityLevel = Field(..., description="Incident severity")
    timestamp: datetime = Field(..., description="Incident occurrence timestamp")
    affected_services: List[str] = Field(default_factory=list, description="Affected services")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class PatternAnalysis(BaseModel):
    """Pattern analysis result."""
    
    pattern_type: str = Field(..., description="Type of pattern detected")
    frequency: int = Field(..., description="Pattern frequency")
    time_window: str = Field(..., description="Time window for pattern")
    examples: List[str] = Field(..., description="Example log entries")
    severity_score: float = Field(..., description="Calculated severity score")


class RootCauseAnalysis(BaseModel):
    """Root cause analysis result."""
    
    primary_cause: str = Field(..., description="Primary root cause")
    contributing_factors: List[str] = Field(..., description="Contributing factors")
    confidence_score: float = Field(..., description="Confidence in analysis")
    evidence: List[str] = Field(..., description="Supporting evidence")
    timeline: List[Dict[str, Any]] = Field(..., description="Timeline of events")


class Recommendation(BaseModel):
    """Actionable recommendation."""
    
    action: str = Field(..., description="Recommended action")
    priority: str = Field(..., description="Priority level")
    estimated_effort: str = Field(..., description="Estimated effort required")
    expected_impact: str = Field(..., description="Expected impact")
    implementation_steps: List[str] = Field(..., description="Implementation steps")


class IncidentAnalysis(BaseModel):
    """Complete incident analysis result."""
    
    incident_id: str = Field(..., description="Incident identifier")
    analysis_timestamp: datetime = Field(..., description="Analysis timestamp")
    log_patterns: List[PatternAnalysis] = Field(..., description="Detected log patterns")
    error_frequency: Dict[str, int] = Field(..., description="Error frequency analysis")
    time_series_data: Dict[str, Any] = Field(..., description="Time series analysis data")
    anomalies: List[Dict[str, Any]] = Field(..., description="Detected anomalies")


class IncidentResponse(BaseModel):
    """Incident analysis response model."""
    
    incident_id: str = Field(..., description="Incident identifier")
    status: IncidentStatus = Field(..., description="Analysis status")
    analysis: IncidentAnalysis = Field(..., description="Incident analysis results")
    root_cause_analysis: RootCauseAnalysis = Field(..., description="Root cause analysis")
    recommendations: List[Recommendation] = Field(..., description="Actionable recommendations")
    summary: str = Field(..., description="Executive summary")
    next_steps: List[str] = Field(..., description="Immediate next steps")
    estimated_resolution_time: Optional[str] = Field(None, description="Estimated resolution time")


class WorkflowState(BaseModel):
    """LangGraph workflow state model."""
    
    incident_request: IncidentRequest
    collected_logs: List[LogEntry] = Field(default_factory=list)
    collected_metrics: List[MetricData] = Field(default_factory=list)
    collected_traces: List[TraceData] = Field(default_factory=list)
    incident_analysis: Optional[IncidentAnalysis] = None
    pattern_analysis: List[PatternAnalysis] = Field(default_factory=list)
    root_cause_analysis: Optional[RootCauseAnalysis] = None
    recommendations: List[Recommendation] = Field(default_factory=list)
    incident_response: Optional[IncidentResponse] = None
    error_messages: List[str] = Field(default_factory=list)


class APIResponse(BaseModel):
    """Generic API response model."""
    
    success: bool = Field(..., description="Request success status")
    message: str = Field(..., description="Response message")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    errors: List[str] = Field(default_factory=list, description="Error messages") 