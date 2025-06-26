"""Main LangGraph workflow for SRE incident management."""

import os
from typing import Dict, Any
from pathlib import Path

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from loguru import logger

from .nodes import WorkflowNodes
from ..data.processors import LogProcessor, MetricsProcessor, TracesProcessor
from ..data.raw_log_processor import RawApacheLogProcessor
from ..models.schemas import WorkflowState, IncidentRequest

# Load environment variables
load_dotenv()


class SREWorkflowAgent:
    """SRE Workflow Agent using LangGraph for incident management."""
    
    def __init__(
        self,
        log_file_path: str = "data/Apache_2k.log",
        templates_file_path: str = "data/Apache_2k.log_templates.csv",
        llm_model: str = "gpt-4o-mini",
        use_raw_logs: bool = None
    ):
        """Initialize the SRE workflow agent."""
        self.log_file_path = log_file_path
        self.templates_file_path = templates_file_path
        self.llm_model = llm_model
        
        # Auto-detect log format if not specified
        if use_raw_logs is None:
            use_raw_logs = log_file_path.endswith('.log')
        self.use_raw_logs = use_raw_logs
        
        # Validate required environment variables
        self._validate_environment()
        
        # Initialize processors based on log format
        if use_raw_logs:
            self.log_processor = RawApacheLogProcessor(log_file_path)
        else:
            self.log_processor = LogProcessor(log_file_path, templates_file_path)
        
        self.metrics_processor = MetricsProcessor()
        self.traces_processor = TracesProcessor()
        
        # Initialize workflow nodes
        self.nodes = WorkflowNodes(
            log_processor=self.log_processor,
            metrics_processor=self.metrics_processor,
            traces_processor=self.traces_processor,
            llm_model=llm_model
        )
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
        
        logger.info(f"SRE Workflow Agent initialized successfully with model: {llm_model}")
    
    def _validate_environment(self) -> None:
        """Validate required environment variables."""
        required_vars = ["OPENAI_API_KEY"]
        optional_vars = ["OPIK_API_KEY", "OPIK_WORKSPACE", "OPIK_PROJECT_NAME"]
        
        missing_required = []
        for var in required_vars:
            if not os.getenv(var):
                missing_required.append(var)
        
        if missing_required:
            raise ValueError(f"Missing required environment variables: {missing_required}")
        
        # Log optional variables status
        for var in optional_vars:
            if os.getenv(var):
                logger.info(f"✅ {var} configured")
            else:
                logger.warning(f"⚠️  {var} not configured - Opik integration disabled")
        
        logger.info("Environment validation completed")
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        # Create the workflow graph
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("incident_ingestion", self.nodes.incident_ingestion_node)
        workflow.add_node("data_collection", self.nodes.data_collection_node)
        workflow.add_node("log_analysis", self.nodes.log_analysis_node)
        workflow.add_node("pattern_recognition", self.nodes.pattern_recognition_node)
        workflow.add_node("rca_generation", self.nodes.rca_generation_node)
        workflow.add_node("recommendation", self.nodes.recommendation_node)
        workflow.add_node("report_generation", self.nodes.report_generation_node)
        
        # Set entry point
        workflow.set_entry_point("incident_ingestion")
        
        # Add edges (workflow sequence)
        workflow.add_edge("incident_ingestion", "data_collection")
        workflow.add_edge("data_collection", "log_analysis")
        workflow.add_edge("log_analysis", "pattern_recognition")
        workflow.add_edge("pattern_recognition", "rca_generation")
        workflow.add_edge("rca_generation", "recommendation")
        workflow.add_edge("recommendation", "report_generation")
        workflow.add_edge("report_generation", END)
        
        # Compile the workflow
        compiled_workflow = workflow.compile()
        
        logger.info("LangGraph workflow compiled successfully")
        return compiled_workflow
    
    async def analyze_incident(self, incident_request: IncidentRequest) -> Dict[str, Any]:
        """Analyze an incident using the workflow."""
        logger.info(f"Starting incident analysis for: {incident_request.incident_id}")
        
        try:
            # Create initial workflow state
            initial_state = WorkflowState(incident_request=incident_request)
            
            # Execute the workflow
            logger.info("Executing LangGraph workflow...")
            result = await self.workflow.ainvoke(initial_state)
            
            # Extract the final response - access state fields properly
            incident_response = result.get("incident_response")
            error_messages = result.get("error_messages", [])
            
            if incident_response:
                logger.info(f"Incident analysis completed successfully for {incident_request.incident_id}")
                return {
                    "success": True,
                    "incident_response": incident_response.model_dump(),
                    "errors": error_messages
                }
            else:
                logger.error("Workflow completed but no incident response generated")
                return {
                    "success": False,
                    "error": "No incident response generated",
                    "errors": error_messages
                }
                
        except Exception as e:
            logger.error(f"Error during incident analysis: {e}")
            return {
                "success": False,
                "error": str(e),
                "errors": [str(e)]
            }
    
    def analyze_incident_sync(self, incident_request: IncidentRequest) -> Dict[str, Any]:
        """Synchronous version of incident analysis."""
        logger.info(f"Starting synchronous incident analysis for: {incident_request.incident_id}")
        
        try:
            # Create initial workflow state
            initial_state = WorkflowState(incident_request=incident_request)
            
            # Execute the workflow synchronously
            logger.info("Executing LangGraph workflow (sync)...")
            result = self.workflow.invoke(initial_state)
            
            # Extract the final response - access state fields properly
            incident_response = result.get("incident_response")
            error_messages = result.get("error_messages", [])
            
            if incident_response:
                logger.info(f"Incident analysis completed successfully for {incident_request.incident_id}")
                return {
                    "success": True,
                    "incident_response": incident_response.model_dump(),
                    "errors": error_messages
                }
            else:
                logger.error("Workflow completed but no incident response generated")
                return {
                    "success": False,
                    "error": "No incident response generated",
                    "errors": error_messages
                }
                
        except Exception as e:
            logger.error(f"Error during incident analysis: {e}")
            return {
                "success": False,
                "error": str(e),
                "errors": [str(e)]
            }
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """Get log statistics for dashboard display."""
        try:
            stats = self.log_processor.get_log_statistics()
            return {
                "success": True,
                "data": stats
            }
        except Exception as e:
            logger.error(f"Error getting log statistics: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_error_analysis(self, timeframe_hours: int = 24) -> Dict[str, Any]:
        """Get error analysis for the specified timeframe."""
        try:
            analysis = self.log_processor.analyze_error_patterns(timeframe_hours=timeframe_hours)
            return {
                "success": True,
                "data": analysis
            }
        except Exception as e:
            logger.error(f"Error getting error analysis: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Health check for the workflow agent."""
        try:
            # Check if log data is loaded (different for raw vs CSV processors)
            if self.use_raw_logs:
                log_status = len(self.log_processor.log_entries) > 0
                templates_status = True  # Raw logs don't use templates
            else:
                log_status = self.log_processor.log_data is not None
                templates_status = self.log_processor.templates_data is not None
            
            # Check workflow compilation
            workflow_status = self.workflow is not None
            
            overall_status = log_status and templates_status and workflow_status
            
            return {
                "success": overall_status,
                "status": "healthy" if overall_status else "unhealthy",
                "components": {
                    "log_processor": "ok" if log_status else "error",
                    "templates_processor": "ok" if templates_status else "error",
                    "workflow": "ok" if workflow_status else "error",
                    "metrics_processor": "ok",
                    "traces_processor": "ok"
                },
                "log_entries": len(self.log_processor.log_entries) if self.use_raw_logs and log_status else (len(self.log_processor.log_data) if log_status else 0),
                "templates": len(self.log_processor.templates_data) if templates_status and not self.use_raw_logs else 0
            }
            
        except Exception as e:
            logger.error(f"Error in health check: {e}")
            return {
                "success": False,
                "status": "error",
                "error": str(e)
            } 