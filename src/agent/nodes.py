"""LangGraph workflow nodes for SRE incident management."""

import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from loguru import logger

# Import Opik if available
try:
    import opik
    from opik import track
    from opik.integrations.langchain import OpikTracer
    OPIK_AVAILABLE = True
    logger.info("✅ Opik integration available")
except ImportError:
    OPIK_AVAILABLE = False
    logger.warning("⚠️  Opik not available - LLM evaluation disabled")

from ..data.processors import LogProcessor, MetricsProcessor, TracesProcessor
from ..models.schemas import (
    IncidentAnalysis,
    IncidentStatus,
    PatternAnalysis,
    Recommendation,
    RootCauseAnalysis,
    WorkflowState,
)

# Load environment variables
load_dotenv()


class WorkflowNodes:
    """LangGraph workflow nodes for SRE incident management."""
    
    def __init__(
        self,
        log_processor: LogProcessor,
        metrics_processor: MetricsProcessor,
        traces_processor: TracesProcessor,
        llm_model: str = "gpt-4o-mini",
    ):
        """Initialize workflow nodes with processors and LLM."""
        self.log_processor = log_processor
        self.metrics_processor = metrics_processor
        self.traces_processor = traces_processor
        
        # Initialize LLM with Opik tracing if available
        self.llm = self._initialize_llm(llm_model)
        
        # Initialize Opik client if available
        self.opik_client = self._initialize_opik()
        
        logger.info(f"WorkflowNodes initialized with model: {llm_model}")
    
    def _initialize_llm(self, model: str) -> ChatOpenAI:
        """Initialize LLM with optional Opik tracing."""
        llm_kwargs = {
            "model": model,
            "temperature": 0.1,
            "max_tokens": 2000
        }
        
        # Add Opik tracing if available and configured
        if OPIK_AVAILABLE and self._is_opik_configured():
            try:
                # Set environment variables for Opik (following quickstart guide)
                import os
                os.environ["OPIK_API_KEY"] = os.getenv("OPIK_API_KEY")
                os.environ["OPIK_WORKSPACE"] = os.getenv("OPIK_WORKSPACE")
                
                tracer = OpikTracer()
                llm_kwargs["callbacks"] = [tracer]
                logger.info("✅ Opik LLM tracing enabled")
            except Exception as e:
                logger.warning(f"⚠️  Failed to initialize Opik tracer: {e}")
        
        return ChatOpenAI(**llm_kwargs)
    
    def _initialize_opik(self) -> Optional[Any]:
        """Initialize Opik client if available and configured."""
        if not OPIK_AVAILABLE:
            return None
        
        try:
            # Initialize Opik client
            client = opik.Opik()
            logger.info("✅ Opik client initialized")
            return client
        except Exception as e:
            logger.warning(f"⚠️  Failed to initialize Opik client: {e}")
            return None
    
    def _is_opik_configured(self) -> bool:
        """Check if Opik is properly configured."""
        required_vars = ["OPIK_API_KEY", "OPIK_WORKSPACE"]
        return all(os.getenv(var) for var in required_vars)
    
    @track(project_name="srenity")
    def incident_ingestion_node(self, state: WorkflowState) -> WorkflowState:
        """Process and validate incoming incident data."""
        logger.info(f"Processing incident: {state.incident_request.incident_id}")
        
        try:
            # Validate incident data
            incident = state.incident_request
            
            # Log incident details
            logger.info(f"Incident '{incident.title}' - Severity: {incident.severity}")
            logger.info(f"Description: {incident.description}")
            logger.info(f"Affected services: {incident.affected_services}")
            
            # Add any pre-processing logic here
            state.error_messages = []
            
            # No need for manual Opik logging here - @track decorator handles it
            
            return state
            
        except Exception as e:
            logger.error(f"Error in incident ingestion: {e}")
            state.error_messages.append(f"Incident ingestion error: {str(e)}")
            return state
    
    @track(project_name="srenity")
    def data_collection_node(self, state: WorkflowState) -> WorkflowState:
        """Collect logs, metrics, and traces data."""
        logger.info("Collecting observability data...")
        
        try:
            incident = state.incident_request
            
            # Define time window around incident
            incident_time = incident.timestamp
            # Ensure timezone-naive datetime for comparison with log data
            if incident_time.tzinfo is not None:
                incident_time = incident_time.replace(tzinfo=None)
            start_time = incident_time - timedelta(hours=2)
            end_time = incident_time + timedelta(hours=1)
            
            # Collect logs
            logger.info("Collecting log data...")
            logs = self.log_processor.get_logs_by_timeframe(start_time, end_time)
            error_logs = self.log_processor.get_error_logs(timeframe_hours=3)
            
            state.collected_logs = logs + error_logs
            logger.info(f"Collected {len(state.collected_logs)} log entries")
            
            # Generate mock metrics
            logger.info("Generating metrics data...")
            metrics = self.metrics_processor.generate_metrics(start_time, end_time)
            state.collected_metrics = metrics
            logger.info(f"Generated {len(metrics)} metric data points")
            
            # Generate mock traces
            logger.info("Generating trace data...")
            traces = self.traces_processor.generate_traces(start_time, end_time)
            state.collected_traces = traces
            logger.info(f"Generated {len(traces)} trace entries")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in data collection: {e}")
            state.error_messages.append(f"Data collection error: {str(e)}")
            return state
    
    @track(project_name="srenity")
    def log_analysis_node(self, state: WorkflowState) -> WorkflowState:
        """Analyze collected log data for patterns and anomalies."""
        logger.info("Analyzing log data...")
        
        try:
            if not state.collected_logs:
                logger.warning("No logs available for analysis")
                return state
            
            # Analyze error patterns
            error_analysis = self.log_processor.analyze_error_patterns(timeframe_hours=3)
            
            # Create incident analysis
            analysis = IncidentAnalysis(
                incident_id=state.incident_request.incident_id,
                analysis_timestamp=datetime.now(),
                log_patterns=[],
                error_frequency=error_analysis.get("event_frequency", {}),
                time_series_data={"hourly_distribution": error_analysis.get("hourly_distribution", {})},
                anomalies=[]
            )
            
            # Create pattern analysis entries
            patterns = error_analysis.get("patterns", {})
            for pattern, frequency in patterns.items():
                severity_score = min(frequency / 10.0, 10.0)  # Scale 0-10
                
                pattern_analysis = PatternAnalysis(
                    pattern_type="error_pattern",
                    frequency=frequency,
                    time_window="3 hours",
                    examples=[pattern],
                    severity_score=severity_score
                )
                analysis.log_patterns.append(pattern_analysis)
                state.pattern_analysis.append(pattern_analysis)
            
            # Add anomaly detection
            if error_analysis.get("total_errors", 0) > 50:
                anomaly = {
                    "type": "high_error_rate",
                    "severity": "high",
                    "description": f"Detected {error_analysis['total_errors']} errors in 3-hour window",
                    "threshold_exceeded": True
                }
                analysis.anomalies.append(anomaly)
            
            state.incident_analysis = analysis
            logger.info(f"Completed log analysis with {len(analysis.log_patterns)} patterns")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in log analysis: {e}")
            state.error_messages.append(f"Log analysis error: {str(e)}")
            return state
    
    @track(project_name="srenity")
    def pattern_recognition_node(self, state: WorkflowState) -> WorkflowState:
        """Use LLM to identify patterns and correlations across all data."""
        logger.info("Performing pattern recognition with LLM...")
        
        try:
            # Prepare data summary for LLM
            data_summary = self._prepare_data_summary(state)
            
            # Use the official @track decorator pattern for LLM calls
            llm_analysis = self._analyze_patterns_with_llm(
                incident_title=state.incident_request.title,
                incident_description=state.incident_request.description,
                incident_severity=str(state.incident_request.severity),
                data_summary=data_summary
            )
            
            # Create additional pattern analysis based on LLM insights
            llm_pattern = PatternAnalysis(
                pattern_type="llm_analysis",
                frequency=1,
                time_window="incident_window",
                examples=[llm_analysis],
                severity_score=5.0  # Default severity
            )
            
            state.pattern_analysis.append(llm_pattern)
            logger.info("Completed pattern recognition with LLM")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in pattern recognition: {e}")
            state.error_messages.append(f"Pattern recognition error: {str(e)}")
            return state
    
    @track(project_name="srenity")
    def rca_generation_node(self, state: WorkflowState) -> WorkflowState:
        """Generate root cause analysis using LLM."""
        logger.info("Generating root cause analysis...")
        
        try:
            # Prepare comprehensive data for RCA
            rca_data = self._prepare_rca_data(state)
            
            # Use the official @track decorator pattern for LLM calls
            rca_text = self._generate_root_cause_analysis_llm(
                incident_title=state.incident_request.title,
                incident_description=state.incident_request.description,
                rca_data=f"{rca_data}\nSeverity: {state.incident_request.severity}\nPattern Analysis: {[p.pattern_type for p in state.pattern_analysis]}"
            )
            
            # Parse RCA into structured format
            rca = self._parse_rca_response(rca_text, state)
            state.root_cause_analysis = rca
            
            logger.info("Completed root cause analysis")
            return state
            
        except Exception as e:
            logger.error(f"Error in RCA generation: {e}")
            state.error_messages.append(f"RCA generation error: {str(e)}")
            return state
    
    def recommendation_node(self, state: WorkflowState) -> WorkflowState:
        """Generate actionable recommendations."""
        logger.info("Generating recommendations...")
        
        try:
            # Prepare recommendation context
            rec_context = self._prepare_recommendation_context(state)
            
            system_prompt = """You are an expert SRE consultant providing actionable recommendations. Based on the incident analysis and root cause, provide:
            1. Immediate actions to resolve the issue
            2. Short-term preventive measures
            3. Long-term improvements
            4. Monitoring and alerting enhancements
            
            Each recommendation should include priority, effort estimate, and expected impact."""
            
            user_prompt = f"""Generate actionable recommendations for this incident:
            
            Context:
            {rec_context}
            
            Provide specific, actionable recommendations with priorities, effort estimates, and implementation steps."""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            recommendations_text = response.content
            
            # Parse recommendations
            recommendations = self._parse_recommendations(recommendations_text)
            state.recommendations = recommendations
            
            logger.info(f"Generated {len(recommendations)} recommendations")
            return state
            
        except Exception as e:
            logger.error(f"Error in recommendation generation: {e}")
            state.error_messages.append(f"Recommendation error: {str(e)}")
            return state
    
    def report_generation_node(self, state: WorkflowState) -> WorkflowState:
        """Generate final incident report."""
        logger.info("Generating incident report...")
        
        try:
            # Create executive summary
            summary = self._generate_executive_summary(state)
            
            # Determine next steps
            next_steps = [
                "Implement immediate recommendations",
                "Monitor system for stability",
                "Review and update incident response procedures",
                "Schedule post-incident review meeting"
            ]
            
            # Estimate resolution time
            if state.incident_request.severity == "critical":
                resolution_time = "2-4 hours"
            elif state.incident_request.severity == "high":
                resolution_time = "4-8 hours"
            else:
                resolution_time = "1-2 days"
            
            # Create a default incident analysis if none exists
            if state.incident_analysis is None:
                from ..models.schemas import IncidentAnalysis
                state.incident_analysis = IncidentAnalysis(
                    incident_id=state.incident_request.incident_id,
                    analysis_timestamp=datetime.now(),
                    log_patterns=[],
                    error_frequency={},
                    time_series_data={},
                    anomalies=[]
                )
            
            # Create default RCA if none exists
            if state.root_cause_analysis is None:
                state.root_cause_analysis = RootCauseAnalysis(
                    primary_cause="Analysis completed with partial data",
                    contributing_factors=["Limited data available for analysis"],
                    confidence_score=0.5,
                    evidence=["Basic incident processing completed"],
                    timeline=[{"timestamp": datetime.now().isoformat(), "event": "Analysis completed"}]
                )
            
            # Create final incident response
            from ..models.schemas import IncidentResponse
            
            incident_response = IncidentResponse(
                incident_id=state.incident_request.incident_id,
                status=IncidentStatus.IN_PROGRESS,
                analysis=state.incident_analysis,
                root_cause_analysis=state.root_cause_analysis,
                recommendations=state.recommendations,
                summary=summary,
                next_steps=next_steps,
                estimated_resolution_time=resolution_time
            )
            
            state.incident_response = incident_response
            logger.info("Incident report generated successfully")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in report generation: {e}")
            state.error_messages.append(f"Report generation error: {str(e)}")
            return state
    
    def _prepare_data_summary(self, state: WorkflowState) -> str:
        """Prepare data summary for LLM analysis."""
        summary_parts = []
        
        # Log summary
        if state.collected_logs:
            error_count = len([log for log in state.collected_logs if log.level.value == "error"])
            summary_parts.append(f"Logs: {len(state.collected_logs)} entries, {error_count} errors")
        
        # Metrics summary
        if state.collected_metrics:
            summary_parts.append(f"Metrics: {len(state.collected_metrics)} data points")
        
        # Traces summary
        if state.collected_traces:
            error_traces = len([trace for trace in state.collected_traces if trace.status == "error"])
            summary_parts.append(f"Traces: {len(state.collected_traces)} traces, {error_traces} errors")
        
        # Pattern analysis summary
        if state.pattern_analysis:
            summary_parts.append(f"Patterns: {len(state.pattern_analysis)} patterns identified")
        
        return "\n".join(summary_parts)
    
    def _prepare_rca_data(self, state: WorkflowState) -> str:
        """Prepare comprehensive data for RCA."""
        rca_parts = []
        
        # Include error patterns
        if state.incident_analysis and state.incident_analysis.error_frequency:
            rca_parts.append(f"Error Frequency: {state.incident_analysis.error_frequency}")
        
        # Include time series data
        if state.incident_analysis and state.incident_analysis.time_series_data:
            rca_parts.append(f"Time Series: {state.incident_analysis.time_series_data}")
        
        # Include anomalies
        if state.incident_analysis and state.incident_analysis.anomalies:
            rca_parts.append(f"Anomalies: {state.incident_analysis.anomalies}")
        
        return "\n".join(rca_parts)
    
    def _prepare_recommendation_context(self, state: WorkflowState) -> str:
        """Prepare context for recommendation generation."""
        context_parts = [
            f"Incident: {state.incident_request.title}",
            f"Severity: {state.incident_request.severity}",
        ]
        
        if state.root_cause_analysis:
            context_parts.append(f"Root Cause: {state.root_cause_analysis.primary_cause}")
            context_parts.append(f"Contributing Factors: {state.root_cause_analysis.contributing_factors}")
        
        return "\n".join(context_parts)
    
    def _parse_rca_response(self, rca_text: str, state: WorkflowState) -> RootCauseAnalysis:
        """Parse LLM RCA response into structured format."""
        # Simple parsing - in production, use more sophisticated parsing
        return RootCauseAnalysis(
            primary_cause="LLM Analysis - See full analysis",
            contributing_factors=["Multiple factors identified"],
            confidence_score=0.7,
            evidence=[rca_text],
            timeline=[{"timestamp": datetime.now().isoformat(), "event": "Analysis completed"}]
        )
    
    def _parse_recommendations(self, rec_text: str) -> List[Recommendation]:
        """Parse recommendations from LLM response."""
        # Simple parsing - in production, use more sophisticated parsing
        return [
            Recommendation(
                action="Implement immediate fixes",
                priority="high",
                estimated_effort="2-4 hours",
                expected_impact="Resolve immediate issue",
                implementation_steps=["Identify root cause", "Apply fix", "Monitor"]
            ),
            Recommendation(
                action="Improve monitoring",
                priority="medium",
                estimated_effort="1-2 days",
                expected_impact="Prevent future incidents",
                implementation_steps=["Review alerts", "Add new monitors", "Test alerting"]
            )
        ]
    
    @track(project_name="srenity")
    def _analyze_patterns_with_llm(self, incident_title: str, incident_description: str, 
                                  incident_severity: str, data_summary: str) -> str:
        """Analyze patterns using LLM with proper Opik tracking."""
        system_prompt = """You are an expert SRE analyst. Analyze the provided observability data to identify patterns, correlations, and potential issues. Focus on:
        1. Error patterns and their frequency
        2. Correlations between logs, metrics, and traces
        3. Anomalies and their potential causes
        4. Time-based patterns
        
        Provide a structured analysis with key findings and potential root causes."""
        
        user_prompt = f"""Analyze this incident data:
        
        Incident: {incident_title}
        Description: {incident_description}
        Severity: {incident_severity}
        
        Data Summary:
        {data_summary}
        
        Provide your analysis in a structured format."""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content

    @track(project_name="srenity")
    def _generate_root_cause_analysis_llm(self, incident_title: str, incident_description: str,
                                         rca_data: str) -> str:
        """Generate root cause analysis using LLM with proper Opik tracking."""
        system_prompt = """You are an expert SRE engineer performing root cause analysis. Based on the incident data and observability information, determine:
        1. The most likely primary root cause
        2. Contributing factors
        3. Supporting evidence from the data
        4. Confidence level in your analysis
        
        Provide a structured RCA with clear reasoning."""
        
        user_prompt = f"""Perform root cause analysis for:
        
        Incident: {incident_title}
        Description: {incident_description}
        
        Analysis Data:
        {rca_data}
        
        Provide detailed RCA in structured format."""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content

    def _generate_executive_summary(self, state: WorkflowState) -> str:
        """Generate executive summary of the incident."""
        summary_parts = [
            f"Incident {state.incident_request.incident_id} - {state.incident_request.title}",
            f"Severity: {state.incident_request.severity.upper()}",
            f"Occurred: {state.incident_request.timestamp}",
        ]
        
        if state.root_cause_analysis:
            summary_parts.append(f"Root Cause: {state.root_cause_analysis.primary_cause}")
        
        if state.recommendations:
            summary_parts.append(f"Recommendations: {len(state.recommendations)} actions identified")
        
        return "\n".join(summary_parts) 