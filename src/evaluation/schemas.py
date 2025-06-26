"""Evaluation schemas for SRE Workflow Agent."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class ScenarioType(str, Enum):
    """Types of evaluation scenarios."""
    
    SLOW_API_ENDPOINT = "slow_api_endpoint"
    DISK_SPACE_ALERT = "disk_space_alert"
    HIGH_ERROR_RATE = "high_error_rate"
    DATABASE_LOCK = "database_lock"
    MEMORY_LEAK = "memory_leak"
    NETWORK_CONNECTIVITY = "network_connectivity"


class EvaluationStatus(str, Enum):
    """Evaluation status types."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class RCAEvaluation(BaseModel):
    """Root Cause Analysis evaluation results."""
    
    precision: float = Field(..., description="Precision score for RCA identification")
    recall: float = Field(..., description="Recall score for RCA identification")
    f1_score: float = Field(..., description="F1 score for RCA identification")
    accuracy: float = Field(..., description="Overall accuracy of RCA")
    predicted_causes: List[str] = Field(..., description="Predicted root causes")
    actual_causes: List[str] = Field(..., description="Actual root causes")
    true_positives: int = Field(..., description="True positive count")
    false_positives: int = Field(..., description="False positive count")
    false_negatives: int = Field(..., description="False negative count")
    confidence_scores: List[float] = Field(..., description="Confidence scores for each prediction")


class CorrelationEvaluation(BaseModel):
    """Event correlation evaluation results."""
    
    temporal_correlation_score: float = Field(..., description="Temporal correlation accuracy")
    correlation_precision: float = Field(..., description="Precision of event correlations")
    correlation_recall: float = Field(..., description="Recall of event correlations")
    event_ordering_accuracy: float = Field(..., description="Accuracy of event ordering")
    cross_service_correlation: float = Field(..., description="Cross-service correlation accuracy")
    correlation_latency_ms: float = Field(..., description="Average correlation processing time")
    detected_correlations: List[Dict[str, Any]] = Field(..., description="Detected event correlations")
    missed_correlations: List[Dict[str, Any]] = Field(..., description="Missed correlations")


class LLMEvaluation(BaseModel):
    """LLM-specific evaluation results."""
    
    hallucination_rate: float = Field(..., description="Rate of hallucinated information")
    coherence_score: float = Field(..., description="Logical flow coherence score")
    grounding_score: float = Field(..., description="Evidence grounding score")
    relevance_score: float = Field(..., description="Response relevance score")
    completeness_score: float = Field(..., description="Response completeness score")
    factual_accuracy: float = Field(..., description="Factual accuracy of responses")
    hallucinated_claims: List[str] = Field(..., description="Identified hallucinated claims")
    ungrounded_statements: List[str] = Field(..., description="Statements not grounded in data")
    response_quality_score: float = Field(..., description="Overall response quality")


class OperationalEvaluation(BaseModel):
    """SRE/Operational metrics evaluation results."""
    
    mttr_seconds: float = Field(..., description="Mean Time to Resolution in seconds")
    mttr_improvement_percent: float = Field(..., description="MTTR improvement percentage")
    incident_resolution_accuracy: float = Field(..., description="Accuracy of incident resolution")
    cost_efficiency_score: float = Field(..., description="Cost efficiency score")
    false_positive_rate: float = Field(..., description="False positive alert rate")
    automation_coverage: float = Field(..., description="Percentage of automated responses")
    escalation_rate: float = Field(..., description="Rate of incident escalations")
    customer_impact_reduction: float = Field(..., description="Reduction in customer impact")


class PerformanceEvaluation(BaseModel):
    """Performance metrics evaluation results."""
    
    analysis_time_seconds: float = Field(..., description="Total analysis time")
    data_processing_time_seconds: float = Field(..., description="Data processing time")
    llm_response_time_seconds: float = Field(..., description="LLM response time")
    memory_usage_mb: float = Field(..., description="Peak memory usage")
    cpu_usage_percent: float = Field(..., description="Average CPU usage")
    throughput_incidents_per_hour: float = Field(..., description="Incident processing throughput")
    api_response_time_ms: float = Field(..., description="API response time")
    scalability_score: float = Field(..., description="System scalability score")


class ScenarioResult(BaseModel):
    """Individual scenario evaluation result."""
    
    scenario_id: str = Field(..., description="Unique scenario identifier")
    scenario_type: ScenarioType = Field(..., description="Type of scenario")
    scenario_name: str = Field(..., description="Scenario name")
    description: str = Field(..., description="Scenario description")
    expected_behavior: str = Field(..., description="Expected system behavior")
    actual_behavior: str = Field(..., description="Actual system behavior")
    success: bool = Field(..., description="Whether scenario passed")
    score: float = Field(..., description="Scenario score (0-1)")
    execution_time_seconds: float = Field(..., description="Scenario execution time")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    detailed_results: Dict[str, Any] = Field(..., description="Detailed evaluation results")


class EvaluationResult(BaseModel):
    """Complete evaluation results."""
    
    evaluation_id: str = Field(..., description="Unique evaluation identifier")
    timestamp: datetime = Field(..., description="Evaluation timestamp")
    status: EvaluationStatus = Field(..., description="Evaluation status")
    overall_score: float = Field(..., description="Overall evaluation score (0-1)")
    
    # Individual evaluation components
    rca_evaluation: RCAEvaluation = Field(..., description="Root Cause Analysis evaluation")
    correlation_evaluation: CorrelationEvaluation = Field(..., description="Event correlation evaluation")
    llm_evaluation: LLMEvaluation = Field(..., description="LLM performance evaluation")
    operational_evaluation: OperationalEvaluation = Field(..., description="Operational metrics evaluation")
    performance_evaluation: PerformanceEvaluation = Field(..., description="Performance metrics evaluation")
    
    # Scenario results
    scenario_results: List[ScenarioResult] = Field(..., description="Individual scenario results")
    scenario_success_rate: float = Field(..., description="Percentage of scenarios passed")
    
    # Summary
    total_scenarios: int = Field(..., description="Total number of scenarios evaluated")
    passed_scenarios: int = Field(..., description="Number of passed scenarios")
    failed_scenarios: int = Field(..., description="Number of failed scenarios")
    
    # Execution details
    total_execution_time_seconds: float = Field(..., description="Total evaluation execution time")
    configuration: Dict[str, Any] = Field(..., description="Evaluation configuration used")
    recommendations: List[str] = Field(..., description="Improvement recommendations")
    error_summary: List[str] = Field(default_factory=list, description="Summary of errors encountered")


class EvaluationConfig(BaseModel):
    """Configuration for evaluation runs."""
    
    # General settings
    evaluation_name: str = Field(..., description="Name of evaluation run")
    description: str = Field(..., description="Description of evaluation")
    
    # Scenario selection
    scenarios_to_run: List[ScenarioType] = Field(..., description="Scenarios to include in evaluation")
    custom_scenarios: List[Dict[str, Any]] = Field(default_factory=list, description="Custom scenario definitions")
    
    # Evaluation parameters
    timeout_seconds: int = Field(default=300, description="Timeout for each scenario")
    parallel_execution: bool = Field(default=False, description="Enable parallel scenario execution")
    
    # RCA evaluation settings
    rca_ground_truth_file: Optional[str] = Field(None, description="Path to RCA ground truth file")
    rca_confidence_threshold: float = Field(default=0.7, description="Confidence threshold for RCA predictions")
    
    # Correlation evaluation settings
    correlation_time_window_seconds: int = Field(default=300, description="Time window for correlation analysis")
    correlation_threshold: float = Field(default=0.8, description="Correlation threshold")
    
    # LLM evaluation settings
    llm_evaluation_model: str = Field(default="gpt-4", description="Model to use for LLM evaluation")
    hallucination_detection_enabled: bool = Field(default=True, description="Enable hallucination detection")
    
    # Performance settings
    performance_baseline_file: Optional[str] = Field(None, description="Path to performance baseline file")
    resource_monitoring_enabled: bool = Field(default=True, description="Enable resource monitoring")
    
    # Output settings
    save_detailed_results: bool = Field(default=True, description="Save detailed results to files")
    output_directory: str = Field(default="evaluation_results", description="Directory for evaluation outputs")
    generate_report: bool = Field(default=True, description="Generate evaluation report")


class GroundTruth(BaseModel):
    """Ground truth data for evaluation."""
    
    scenario_id: str = Field(..., description="Scenario identifier")
    incident_id: str = Field(..., description="Incident identifier")
    
    # RCA ground truth
    true_root_causes: List[str] = Field(..., description="True root causes")
    contributing_factors: List[str] = Field(default_factory=list, description="Contributing factors")
    
    # Correlation ground truth
    expected_correlations: List[Dict[str, Any]] = Field(..., description="Expected event correlations")
    event_timeline: List[Dict[str, Any]] = Field(..., description="True event timeline")
    
    # Resolution ground truth
    expected_resolution_time_seconds: float = Field(..., description="Expected resolution time")
    expected_actions: List[str] = Field(..., description="Expected resolution actions")
    
    # Quality ground truth
    quality_score: float = Field(..., description="Expected quality score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata") 