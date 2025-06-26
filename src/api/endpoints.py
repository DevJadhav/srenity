"""FastAPI endpoints for SRE workflow agent."""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse
from loguru import logger
import os

try:
    # Try relative imports first (when run as module)
    from ..agent.workflow import SREWorkflowAgent
    from ..models.schemas import (
        IncidentRequest, 
        IncidentResponse, 
        APIResponse,
        LogEntry,
        MetricData,
        TraceData
    )
    from ..evaluation.evaluator import SRenityEvaluator
    from ..evaluation.schemas import EvaluationConfig, ScenarioType
except ImportError:
    # Fall back to absolute imports (when run directly)
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agent.workflow import SREWorkflowAgent
    from models.schemas import (
        IncidentRequest, 
        IncidentResponse, 
        APIResponse,
        LogEntry,
        MetricData,
        TraceData
    )
    from evaluation.evaluator import SRenityEvaluator
    from evaluation.schemas import EvaluationConfig, ScenarioType

# Initialize router
router = APIRouter()

# Global agent instance (will be initialized in main.py)
agent: Optional[SREWorkflowAgent] = None

# Initialize the evaluator (will be set when agent is available)
evaluator = None

def set_agent(sre_agent: SREWorkflowAgent):
    """Set the global agent instance."""
    global agent, evaluator
    agent = sre_agent
    evaluator = SRenityEvaluator(sre_agent)


@router.post("/incidents/analyze", response_model=APIResponse)
async def analyze_incident(incident_request: IncidentRequest) -> APIResponse:
    """Analyze an incident using the SRE workflow agent."""
    if not agent:
        raise HTTPException(status_code=500, detail="SRE agent not initialized")
    
    try:
        logger.info(f"Received incident analysis request: {incident_request.incident_id}")
        
        # Analyze the incident
        result = agent.analyze_incident_sync(incident_request)
        
        if result["success"]:
            return APIResponse(
                success=True,
                message="Incident analysis completed successfully",
                data=result["incident_response"],
                errors=result.get("errors", [])
            )
        else:
            return APIResponse(
                success=False,
                message="Incident analysis failed",
                data=None,
                errors=result.get("errors", [result.get("error", "Unknown error")])
            )
    
    except Exception as e:
        logger.error(f"Error processing incident analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs", response_model=APIResponse)
async def get_logs(
    start_time: Optional[datetime] = Query(None, description="Start time for log retrieval"),
    end_time: Optional[datetime] = Query(None, description="End time for log retrieval"),
    level: Optional[str] = Query(None, description="Log level filter"),
    limit: int = Query(100, description="Maximum number of logs to return")
) -> APIResponse:
    """Get logs from the system."""
    if not agent:
        raise HTTPException(status_code=500, detail="SRE agent not initialized")
    
    try:
        # Set default time range if not provided
        if not end_time:
            end_time = datetime.now()
        if not start_time:
            start_time = end_time - timedelta(hours=24)
        
        # Get logs from the processor
        logs = agent.log_processor.get_logs_by_timeframe(start_time, end_time)
        
        # Filter by level if specified
        if level:
            logs = [log for log in logs if log.level.value.lower() == level.lower()]
        
        # Limit results
        logs = logs[:limit]
        
        # Convert to dict for response
        logs_data = [log.model_dump() for log in logs]
        
        return APIResponse(
            success=True,
            message=f"Retrieved {len(logs_data)} log entries",
            data={
                "logs": logs_data,
                "count": len(logs_data),
                "time_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat()
                }
            }
        )
    
    except Exception as e:
        logger.error(f"Error retrieving logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics", response_model=APIResponse)
async def get_metrics(
    start_time: Optional[datetime] = Query(None, description="Start time for metrics"),
    end_time: Optional[datetime] = Query(None, description="End time for metrics"),
    metric_name: Optional[str] = Query(None, description="Specific metric name"),
    interval_minutes: int = Query(5, description="Interval between metrics in minutes")
) -> APIResponse:
    """Get system metrics."""
    if not agent:
        raise HTTPException(status_code=500, detail="SRE agent not initialized")
    
    try:
        # Set default time range if not provided
        if not end_time:
            end_time = datetime.now()
        if not start_time:
            start_time = end_time - timedelta(hours=2)
        
        # Generate metrics
        metrics = agent.metrics_processor.generate_metrics(
            start_time, 
            end_time, 
            interval_minutes
        )
        
        # Filter by metric name if specified
        if metric_name:
            metrics = [m for m in metrics if m.metric_name == metric_name]
        
        # Convert to dict for response
        metrics_data = [metric.model_dump() for metric in metrics]
        
        # Analyze for anomalies
        anomalies = agent.metrics_processor.analyze_metric_anomalies(metrics)
        
        return APIResponse(
            success=True,
            message=f"Retrieved {len(metrics_data)} metric data points",
            data={
                "metrics": metrics_data,
                "count": len(metrics_data),
                "anomalies": anomalies,
                "time_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat()
                }
            }
        )
    
    except Exception as e:
        logger.error(f"Error retrieving metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/traces", response_model=APIResponse)
async def get_traces(
    start_time: Optional[datetime] = Query(None, description="Start time for traces"),
    end_time: Optional[datetime] = Query(None, description="End time for traces"),
    operation: Optional[str] = Query(None, description="Filter by operation name"),
    status: Optional[str] = Query(None, description="Filter by trace status"),
    limit: int = Query(100, description="Maximum number of traces to return")
) -> APIResponse:
    """Get system traces."""
    if not agent:
        raise HTTPException(status_code=500, detail="SRE agent not initialized")
    
    try:
        # Set default time range if not provided
        if not end_time:
            end_time = datetime.now()
        if not start_time:
            start_time = end_time - timedelta(hours=2)
        
        # Generate traces
        traces = agent.traces_processor.generate_traces(start_time, end_time)
        
        # Apply filters
        if operation:
            traces = [t for t in traces if t.operation_name == operation]
        
        if status:
            traces = [t for t in traces if t.status == status]
        
        # Limit results
        traces = traces[:limit]
        
        # Convert to dict for response
        traces_data = [trace.model_dump() for trace in traces]
        
        # Analyze trace performance
        performance_analysis = agent.traces_processor.analyze_trace_performance(traces)
        
        return APIResponse(
            success=True,
            message=f"Retrieved {len(traces_data)} trace entries",
            data={
                "traces": traces_data,
                "count": len(traces_data),
                "performance_analysis": performance_analysis,
                "time_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat()
                }
            }
        )
    
    except Exception as e:
        logger.error(f"Error retrieving traces: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics", response_model=APIResponse)
async def get_statistics() -> APIResponse:
    """Get system statistics."""
    if not agent:
        raise HTTPException(status_code=500, detail="SRE agent not initialized")
    
    try:
        # Get log statistics
        result = agent.get_log_statistics()
        
        if result["success"]:
            return APIResponse(
                success=True,
                message="Statistics retrieved successfully",
                data=result["data"]
            )
        else:
            return APIResponse(
                success=False,
                message="Failed to retrieve statistics",
                data=None,
                errors=[result.get("error", "Unknown error")]
            )
    
    except Exception as e:
        logger.error(f"Error retrieving statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/errors/analysis", response_model=APIResponse)
async def get_error_analysis(
    timeframe_hours: int = Query(24, description="Timeframe in hours for error analysis")
) -> APIResponse:
    """Get error analysis for the specified timeframe."""
    if not agent:
        raise HTTPException(status_code=500, detail="SRE agent not initialized")
    
    try:
        result = agent.get_error_analysis(timeframe_hours)
        
        if result["success"]:
            return APIResponse(
                success=True,
                message="Error analysis completed successfully",
                data=result["data"]
            )
        else:
            return APIResponse(
                success=False,
                message="Error analysis failed",
                data=None,
                errors=[result.get("error", "Unknown error")]
            )
    
    except Exception as e:
        logger.error(f"Error performing error analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=APIResponse)
async def health_check() -> APIResponse:
    """Health check endpoint."""
    if not agent:
        return APIResponse(
            success=False,
            message="SRE agent not initialized",
            data={"status": "unhealthy", "reason": "agent_not_initialized"}
        )
    
    try:
        result = agent.health_check()
        
        return APIResponse(
            success=result["success"],
            message=f"System status: {result['status']}",
            data=result
        )
    
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return APIResponse(
            success=False,
            message="Health check failed",
            data={"status": "error", "reason": str(e)}
        )


@router.get("/incidents/{incident_id}/status", response_model=APIResponse)
async def get_incident_status(incident_id: str) -> APIResponse:
    """Get the status of a specific incident."""
    # This is a placeholder - in a real system, you'd track incident status
    # For now, return a mock response
    return APIResponse(
        success=True,
        message="Incident status retrieved",
        data={
            "incident_id": incident_id,
            "status": "in_progress",
            "last_updated": datetime.now().isoformat(),
            "progress": {
                "data_collection": "completed",
                "log_analysis": "completed", 
                "pattern_recognition": "completed",
                "rca_generation": "completed",
                "recommendations": "completed",
                "report_generation": "completed"
            }
        }
    )


@router.get("/dashboard/summary", response_model=APIResponse)
async def get_dashboard_summary() -> APIResponse:
    """Get summary data for dashboard display."""
    if not agent:
        raise HTTPException(status_code=500, detail="SRE agent not initialized")
    
    try:
        # Get various statistics
        log_stats = agent.get_log_statistics()
        error_analysis = agent.get_error_analysis(24)
        health_status = agent.health_check()
        
        # Current time
        current_time = datetime.now()
        
        # Generate recent metrics sample
        recent_metrics = agent.metrics_processor.generate_metrics(
            current_time - timedelta(hours=1),
            current_time,
            interval_minutes=5
        )
        
        # Count error metrics
        error_rate_metrics = [m for m in recent_metrics if m.metric_name == "error_rate_percent"]
        avg_error_rate = sum(m.value for m in error_rate_metrics) / len(error_rate_metrics) if error_rate_metrics else 0
        
        summary_data = {
            "timestamp": current_time.isoformat(),
            "system_health": health_status.get("status", "unknown"),
            "total_log_entries": log_stats["data"]["total_entries"] if log_stats["success"] else 0,
            "error_count_24h": error_analysis["data"]["total_errors"] if error_analysis["success"] else 0,
            "avg_error_rate": round(avg_error_rate, 2),
            "active_incidents": 0,  # Placeholder
            "recent_patterns": len(error_analysis["data"]["patterns"]) if error_analysis["success"] else 0,
            "log_levels": log_stats["data"]["level_distribution"] if log_stats["success"] else {},
            "top_error_patterns": error_analysis["data"]["patterns"] if error_analysis["success"] else {}
        }
        
        return APIResponse(
            success=True,
            message="Dashboard summary retrieved successfully",
            data=summary_data
        )
    
    except Exception as e:
        logger.error(f"Error generating dashboard summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# New Evaluation Endpoints

@router.post("/evaluation/run", response_model=APIResponse)
async def run_evaluation(config: EvaluationConfig):
    """Run a complete evaluation of the SRE workflow agent."""
    try:
        if not evaluator:
            raise HTTPException(status_code=503, detail="Evaluator not initialized. Please ensure the agent is running.")
        
        logger.info(f"Starting evaluation: {config.evaluation_name}")
        
        # Run the evaluation
        result = await evaluator.run_evaluation(config)
        
        return APIResponse(
            success=True,
            message=f"Evaluation '{config.evaluation_name}' completed successfully",
            data=result.model_dump()
        )
        
    except Exception as e:
        logger.error(f"Error running evaluation: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@router.get("/evaluation/scenarios", response_model=APIResponse)
async def get_evaluation_scenarios():
    """Get all available evaluation scenarios."""
    try:
        if not evaluator:
            raise HTTPException(status_code=503, detail="Evaluator not initialized. Please ensure the agent is running.")
        
        scenarios = evaluator.scenarios.get_all_scenarios()
        
        scenario_info = {}
        for scenario_type, scenario_data in scenarios.items():
            scenario_info[scenario_type] = {
                "name": scenario_data.get("scenario_name"),
                "description": scenario_data.get("description"),
                "expected_behavior": scenario_data.get("expected_behavior")
            }
        
        return APIResponse(
            success=True,
            message="Evaluation scenarios retrieved successfully",
            data=scenario_info
        )
        
    except Exception as e:
        logger.error(f"Error retrieving scenarios: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/evaluation/scenarios/{scenario_type}", response_model=APIResponse)
async def get_evaluation_scenario(scenario_type: ScenarioType):
    """Get details for a specific evaluation scenario."""
    try:
        if not evaluator:
            raise HTTPException(status_code=503, detail="Evaluator not initialized. Please ensure the agent is running.")
        
        scenario = evaluator.scenarios.get_scenario(scenario_type)
        
        if not scenario:
            raise HTTPException(status_code=404, detail=f"Scenario {scenario_type} not found")
        
        # Return scenario without sensitive data like ground truth
        scenario_info = {
            "scenario_name": scenario.get("scenario_name"),
            "description": scenario.get("description"),
            "expected_behavior": scenario.get("expected_behavior"),
            "incident_request": scenario["incident_request"].model_dump(),
            "scenario_type": scenario.get("scenario_type")
        }
        
        return APIResponse(
            success=True,
            message=f"Scenario {scenario_type} retrieved successfully",
            data=scenario_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving scenario: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/evaluation/quick-test", response_model=APIResponse)
async def run_quick_evaluation():
    """Run a quick evaluation with default scenarios."""
    try:
        if not evaluator:
            raise HTTPException(status_code=503, detail="Evaluator not initialized. Please ensure the agent is running.")
        
        logger.info("Starting quick evaluation")
        
        # Create a quick evaluation config
        config = EvaluationConfig(
            evaluation_name="Quick Test",
            description="Quick evaluation with core scenarios",
            scenarios_to_run=[
                ScenarioType.SLOW_API_ENDPOINT,
                ScenarioType.HIGH_ERROR_RATE
            ],
            timeout_seconds=120,
            parallel_execution=True,
            save_detailed_results=False,
            generate_report=False
        )
        
        # Run the evaluation
        result = await evaluator.run_evaluation(config)
        
        # Return simplified results
        simplified_result = {
            "evaluation_id": result.evaluation_id,
            "overall_score": result.overall_score,
            "scenario_success_rate": result.scenario_success_rate,
            "total_scenarios": result.total_scenarios,
            "passed_scenarios": result.passed_scenarios,
            "failed_scenarios": result.failed_scenarios,
            "execution_time": result.total_execution_time_seconds,
            "key_metrics": {
                "rca_f1_score": result.rca_evaluation.f1_score,
                "correlation_score": result.correlation_evaluation.temporal_correlation_score,
                "llm_quality_score": result.llm_evaluation.response_quality_score
            },
            "recommendations": result.recommendations[:3]  # Top 3 recommendations
        }
        
        return APIResponse(
            success=True,
            message="Quick evaluation completed successfully",
            data=simplified_result
        )
        
    except Exception as e:
        logger.error(f"Error running quick evaluation: {e}")
        raise HTTPException(status_code=500, detail=f"Quick evaluation failed: {str(e)}")


@router.get("/evaluation/metrics/definitions", response_model=APIResponse)
async def get_evaluation_metrics_definitions():
    """Get definitions and explanations for all evaluation metrics."""
    try:
        metrics_definitions = {
            "rca_metrics": {
                "precision": "Proportion of predicted root causes that are correct",
                "recall": "Proportion of actual root causes that were predicted",
                "f1_score": "Harmonic mean of precision and recall",
                "accuracy": "Overall accuracy of root cause identification"
            },
            "correlation_metrics": {
                "temporal_correlation_score": "Accuracy of correlating events in the correct time order",
                "correlation_precision": "Precision of event correlations detected",
                "correlation_recall": "Recall of event correlations detected",
                "event_ordering_accuracy": "Accuracy of reconstructing event timeline"
            },
            "llm_metrics": {
                "hallucination_rate": "Rate of claims not supported by available data",
                "coherence_score": "Logical flow and consistency of the response",
                "grounding_score": "How well responses are grounded in evidence",
                "relevance_score": "Relevance of response to the incident",
                "completeness_score": "Completeness of the analysis and recommendations"
            },
            "operational_metrics": {
                "mttr_seconds": "Mean Time to Resolution in seconds",
                "incident_resolution_accuracy": "Accuracy of incident resolution",
                "cost_efficiency_score": "Cost efficiency of incident handling",
                "false_positive_rate": "Rate of false positive alerts"
            },
            "performance_metrics": {
                "analysis_time_seconds": "Time taken for incident analysis",
                "memory_usage_mb": "Peak memory usage during analysis",
                "cpu_usage_percent": "Average CPU usage during analysis",
                "throughput_incidents_per_hour": "Number of incidents processed per hour"
            }
        }
        
        return APIResponse(
            success=True,
            message="Evaluation metrics definitions retrieved successfully",
            data=metrics_definitions
        )
        
    except Exception as e:
        logger.error(f"Error retrieving metrics definitions: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    """Serve the incident management dashboard."""
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    try:
        with open(dashboard_path, "r") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dashboard not found")


@router.get("/workflow/visualize", tags=["workflow"])
async def visualize_workflow(format: str = "mermaid", log_to_opik: bool = False):
    """
    Visualize the workflow graph.
    
    Args:
        format: Output format - 'mermaid' for text diagram or 'png' for image
        log_to_opik: Whether to log the workflow graph to Opik
    
    Returns:
        Workflow visualization in requested format
    """
    try:
        agent = get_workflow_agent()
        
        # Log to Opik if requested
        if log_to_opik and hasattr(agent, 'log_workflow_graph_to_opik'):
            agent.log_workflow_graph_to_opik()
            logger.info("Workflow graph logged to Opik")
        
        if format == "mermaid":
            # Return Mermaid diagram text
            diagram = agent.get_workflow_mermaid()
            return {
                "success": True,
                "format": "mermaid",
                "diagram": diagram,
                "opik_logged": log_to_opik
            }
        elif format == "png":
            # Generate PNG and return path
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                path = agent.visualize_workflow(tmp.name)
                
                # Read the PNG file
                with open(path, "rb") as f:
                    png_data = f.read()
                
                # Convert to base64 for API response
                import base64
                png_base64 = base64.b64encode(png_data).decode()
                
                return {
                    "success": True,
                    "format": "png",
                    "data": png_base64,
                    "mime_type": "image/png",
                    "opik_logged": log_to_opik
                }
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format: {format}. Use 'mermaid' or 'png'"
            )
            
    except Exception as e:
        logger.error(f"Error visualizing workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 