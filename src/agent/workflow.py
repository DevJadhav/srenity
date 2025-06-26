"""Main LangGraph workflow for SRE incident management."""

import os
from typing import Dict, Any
from pathlib import Path
import tempfile
from datetime import datetime
import json

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from loguru import logger

# Import Opik for complete workflow tracing
try:
    import opik
    from opik import track
    from opik.integrations.langchain import OpikTracer
    OPIK_AVAILABLE = True
    logger.info("✅ Opik integration available for workflow tracing")
except ImportError:
    OPIK_AVAILABLE = False
    logger.warning("⚠️  Opik not available - workflow tracing disabled")

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
        use_raw_logs: bool = None,
        enable_opik_tracing: bool = True
    ):
        """Initialize the SRE workflow agent."""
        self.log_file_path = log_file_path
        self.templates_file_path = templates_file_path
        self.llm_model = llm_model
        self.enable_opik_tracing = enable_opik_tracing and OPIK_AVAILABLE
        
        # Auto-detect log format if not specified
        if use_raw_logs is None:
            use_raw_logs = log_file_path.endswith('.log')
        self.use_raw_logs = use_raw_logs
        
        # Validate required environment variables
        self._validate_environment()
        
        # Initialize Opik client if available
        self.opik_client = self._initialize_opik()
        
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
        if self.enable_opik_tracing:
            logger.info("✅ Opik workflow tracing enabled")
    
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
    
    def _initialize_opik(self) -> Any:
        """Initialize Opik client for workflow tracing."""
        if not self.enable_opik_tracing:
            return None
        
        try:
            # Set environment variables for Opik
            os.environ["OPIK_API_KEY"] = os.getenv("OPIK_API_KEY", "")
            os.environ["OPIK_WORKSPACE"] = os.getenv("OPIK_WORKSPACE", "")
            
            # Initialize Opik client
            client = opik.Opik()
            
            # Configure project
            project_name = os.getenv("OPIK_PROJECT_NAME", "srenity")
            logger.info(f"✅ Opik client initialized for project: {project_name}")
            
            return client
        except Exception as e:
            logger.warning(f"⚠️  Failed to initialize Opik client: {e}")
            self.enable_opik_tracing = False
            return None
    
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
    
    def visualize_workflow(self, output_path: str = None) -> str:
        """Visualize the workflow graph using LangGraph's built-in visualization."""
        try:
            # Get workflow graph
            graph = self.workflow.get_graph()
            
            # Generate visualization
            png_data = graph.draw_mermaid_png()
            
            # Save to file
            if output_path is None:
                output_path = "workflow_graph.png"
            
            with open(output_path, "wb") as f:
                f.write(png_data)
            
            logger.info(f"Workflow graph saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error visualizing workflow: {e}")
            # Try alternative visualization method
            return self._generate_mermaid_diagram()
    
    def get_workflow_mermaid(self) -> str:
        """Get the workflow as a Mermaid diagram string."""
        try:
            graph = self.workflow.get_graph()
            return graph.draw_mermaid()
        except Exception as e:
            logger.error(f"Error generating Mermaid diagram: {e}")
            # Fallback to manual generation
            return self._generate_mermaid_diagram()
    
    def _generate_mermaid_diagram(self) -> str:
        """Generate a Mermaid diagram representation of the workflow."""
        mermaid = """graph TD
    A[Incident Ingestion] --> B[Data Collection]
    B --> C[Log Analysis]
    C --> D[Pattern Recognition]
    D --> E[RCA Generation]
    E --> F[Recommendation]
    F --> G[Report Generation]
    G --> H[End]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333,stroke-width:2px
    style C fill:#bbf,stroke:#333,stroke-width:2px
    style D fill:#bbf,stroke:#333,stroke-width:2px
    style E fill:#fbf,stroke:#333,stroke-width:2px
    style F fill:#bfb,stroke:#333,stroke-width:2px
    style G fill:#ffb,stroke:#333,stroke-width:2px
    style H fill:#f9f,stroke:#333,stroke-width:2px
"""
        return mermaid
    
    @track(project_name="srenity", capture_input=True, capture_output=True)
    async def analyze_incident(self, incident_request: IncidentRequest) -> Dict[str, Any]:
        """Analyze an incident using the workflow with complete Opik tracing."""
        logger.info(f"Starting incident analysis for: {incident_request.incident_id}")
        
        # Start Opik trace if available
        trace_data = None
        if self.enable_opik_tracing and self.opik_client:
            trace_data = {
                "name": f"incident_analysis_{incident_request.incident_id}",
                "type": "workflow",
                "metadata": {
                    "incident_id": incident_request.incident_id,
                    "severity": incident_request.severity,
                    "title": incident_request.title,
                    "services": incident_request.affected_services,
                    "workflow_version": "1.0"
                }
            }
        
        try:
            # Create initial workflow state
            initial_state = WorkflowState(incident_request=incident_request)
            
            # Execute the workflow
            logger.info("Executing LangGraph workflow...")
            result = await self.workflow.ainvoke(initial_state)
            
            # Extract the final response
            incident_response = result.get("incident_response")
            error_messages = result.get("error_messages", [])
            
            if incident_response:
                logger.info(f"Incident analysis completed successfully for {incident_request.incident_id}")
                
                # Log to Opik if available
                if self.enable_opik_tracing and trace_data:
                    self._log_workflow_to_opik(trace_data, result, success=True)
                
                return {
                    "success": True,
                    "incident_response": incident_response.model_dump(),
                    "errors": error_messages
                }
            else:
                logger.error("Workflow completed but no incident response generated")
                
                # Log failure to Opik
                if self.enable_opik_tracing and trace_data:
                    self._log_workflow_to_opik(trace_data, result, success=False)
                
                return {
                    "success": False,
                    "error": "No incident response generated",
                    "errors": error_messages
                }
                
        except Exception as e:
            logger.error(f"Error during incident analysis: {e}")
            
            # Log exception to Opik
            if self.enable_opik_tracing and trace_data:
                trace_data["error"] = str(e)
                self._log_workflow_to_opik(trace_data, None, success=False)
            
            return {
                "success": False,
                "error": str(e),
                "errors": [str(e)]
            }
    
    def _log_workflow_to_opik(self, trace_data: Dict[str, Any], result: Any, success: bool) -> None:
        """Log workflow execution to Opik."""
        try:
            if not self.opik_client:
                return
            
            # Add execution result
            trace_data["success"] = success
            if result:
                trace_data["result"] = {
                    "has_response": result.get("incident_response") is not None,
                    "error_count": len(result.get("error_messages", [])),
                    "node_states": list(result.keys())
                }
            
            # Log custom metrics
            if success and result and result.get("incident_response"):
                response = result["incident_response"]
                trace_data["metrics"] = {
                    "pattern_count": len(response.analysis.log_patterns) if hasattr(response, 'analysis') else 0,
                    "recommendation_count": len(response.recommendations) if hasattr(response, 'recommendations') else 0,
                    "confidence_score": response.root_cause_analysis.confidence_score if hasattr(response, 'root_cause_analysis') else 0
                }
            
            # Add workflow graph structure
            workflow_structure = self._get_workflow_structure()
            trace_data["workflow_graph"] = workflow_structure
            
            # Add Mermaid diagram as a separate field for easy access
            trace_data["mermaid_diagram"] = workflow_structure.get("mermaid_diagram", "")
            trace_data["mermaid_live_url"] = "https://mermaid.live"
            
            logger.debug(f"Logged workflow execution to Opik: {trace_data.get('name')}")
            
        except Exception as e:
            logger.warning(f"Failed to log to Opik: {e}")
    
    def _get_workflow_structure(self) -> Dict[str, Any]:
        """Get workflow structure for Opik logging."""
        try:
            # Get Mermaid diagram
            mermaid_diagram = self.get_workflow_mermaid()
            
            # Define workflow nodes and edges
            nodes = [
                {"id": "incident_ingestion", "label": "Incident Ingestion", "type": "input"},
                {"id": "data_collection", "label": "Data Collection", "type": "process"},
                {"id": "log_analysis", "label": "Log Analysis", "type": "process"},
                {"id": "pattern_recognition", "label": "Pattern Recognition", "type": "llm"},
                {"id": "rca_generation", "label": "RCA Generation", "type": "llm"},
                {"id": "recommendation", "label": "Recommendation", "type": "llm"},
                {"id": "report_generation", "label": "Report Generation", "type": "output"}
            ]
            
            edges = [
                {"from": "incident_ingestion", "to": "data_collection"},
                {"from": "data_collection", "to": "log_analysis"},
                {"from": "log_analysis", "to": "pattern_recognition"},
                {"from": "pattern_recognition", "to": "rca_generation"},
                {"from": "rca_generation", "to": "recommendation"},
                {"from": "recommendation", "to": "report_generation"}
            ]
            
            return {
                "mermaid_diagram": mermaid_diagram,
                "nodes": nodes,
                "edges": edges,
                "node_count": len(nodes),
                "edge_count": len(edges),
                "workflow_type": "sequential",
                "version": "1.0"
            }
        except Exception as e:
            logger.warning(f"Failed to get workflow structure: {e}")
            return {"error": str(e)}
    
    @track(project_name="srenity", capture_input=False, capture_output=False)
    def log_workflow_graph_to_opik(self) -> None:
        """Log the workflow graph structure to Opik as a standalone trace."""
        if not self.enable_opik_tracing or not self.opik_client:
            logger.warning("Opik not available for workflow graph logging")
            return
        
        try:
            # Get workflow structure
            workflow_structure = self._get_workflow_structure()
            
            # Create a special trace for the workflow graph
            trace_metadata = {
                "type": "workflow_graph",
                "name": "SRenity Workflow Graph",
                "description": "LangGraph workflow structure for SRE incident management",
                "timestamp": datetime.now().isoformat(),
                "workflow_structure": workflow_structure,
                "visualization": {
                    "mermaid": workflow_structure.get("mermaid_diagram", ""),
                    "nodes": workflow_structure.get("nodes", []),
                    "edges": workflow_structure.get("edges", []),
                    "rendering_instructions": "Copy the mermaid diagram to https://mermaid.live for visualization"
                }
            }
            
            # Log workflow metadata using track context
            # The metadata will be automatically attached to the current trace
            logger.info(f"Workflow Graph Structure: {json.dumps(trace_metadata, indent=2)}")
            
            # Log each node as a span for visualization
            for i, node in enumerate(workflow_structure.get("nodes", [])):
                self._log_workflow_node_to_opik(node, i)
            
            logger.info("✅ Workflow graph logged to Opik")
            
        except Exception as e:
            logger.error(f"Failed to log workflow graph to Opik: {e}")
    
    @track(project_name="srenity", name="workflow_node", capture_input=True, capture_output=True)
    def _log_workflow_node_to_opik(self, node: Dict[str, Any], order: int) -> Dict[str, Any]:
        """Log individual workflow node to Opik."""
        # This creates a span for each workflow node with metadata
        node_info = {
            "node_id": node.get("id"),
            "node_label": node.get("label"),
            "node_type": node.get("type"),
            "execution_order": order
        }
        logger.debug(f"Workflow Node: {node_info}")
        return node_info
    
    def visualize_and_log_workflow(self, output_path: str = None) -> str:
        """Visualize workflow and log it to Opik."""
        # First generate the visualization
        result = self.visualize_workflow(output_path)
        
        # Then log to Opik
        if self.enable_opik_tracing:
            self.log_workflow_graph_to_opik()
        
        return result
    
    @track(project_name="srenity", capture_input=True, capture_output=True)
    def analyze_incident_sync(self, incident_request: IncidentRequest) -> Dict[str, Any]:
        """Synchronous version of incident analysis with Opik tracing."""
        logger.info(f"Starting synchronous incident analysis for: {incident_request.incident_id}")
        
        # Start Opik trace if available
        trace_data = None
        if self.enable_opik_tracing and self.opik_client:
            trace_data = {
                "name": f"incident_analysis_sync_{incident_request.incident_id}",
                "type": "workflow",
                "metadata": {
                    "incident_id": incident_request.incident_id,
                    "severity": incident_request.severity,
                    "title": incident_request.title,
                    "services": incident_request.affected_services,
                    "workflow_version": "1.0",
                    "sync_mode": True
                }
            }
        
        try:
            # Create initial workflow state
            initial_state = WorkflowState(incident_request=incident_request)
            
            # Execute the workflow synchronously
            logger.info("Executing LangGraph workflow (sync)...")
            result = self.workflow.invoke(initial_state)
            
            # Extract the final response
            incident_response = result.get("incident_response")
            error_messages = result.get("error_messages", [])
            
            if incident_response:
                logger.info(f"Incident analysis completed successfully for {incident_request.incident_id}")
                
                # Log to Opik if available
                if self.enable_opik_tracing and trace_data:
                    self._log_workflow_to_opik(trace_data, result, success=True)
                
                return {
                    "success": True,
                    "incident_response": incident_response.model_dump(),
                    "errors": error_messages
                }
            else:
                logger.error("Workflow completed but no incident response generated")
                
                # Log failure to Opik
                if self.enable_opik_tracing and trace_data:
                    self._log_workflow_to_opik(trace_data, result, success=False)
                
                return {
                    "success": False,
                    "error": "No incident response generated",
                    "errors": error_messages
                }
                
        except Exception as e:
            logger.error(f"Error during incident analysis: {e}")
            
            # Log exception to Opik
            if self.enable_opik_tracing and trace_data:
                trace_data["error"] = str(e)
                self._log_workflow_to_opik(trace_data, None, success=False)
            
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
            # Check if log data is loaded
            if self.use_raw_logs:
                log_status = len(self.log_processor.log_entries) > 0
                templates_status = True  # Raw logs don't use templates
            else:
                log_status = self.log_processor.log_data is not None
                templates_status = self.log_processor.templates_data is not None
            
            # Check workflow compilation
            workflow_status = self.workflow is not None
            
            # Check Opik integration
            opik_status = self.enable_opik_tracing and self.opik_client is not None
            
            overall_status = log_status and templates_status and workflow_status
            
            return {
                "success": overall_status,
                "status": "healthy" if overall_status else "unhealthy",
                "components": {
                    "log_processor": "ok" if log_status else "error",
                    "templates_processor": "ok" if templates_status else "error",
                    "workflow": "ok" if workflow_status else "error",
                    "metrics_processor": "ok",
                    "traces_processor": "ok",
                    "opik_integration": "ok" if opik_status else "disabled"
                },
                "log_entries": len(self.log_processor.log_entries) if self.use_raw_logs and log_status else (len(self.log_processor.log_data) if log_status else 0),
                "templates": len(self.log_processor.templates_data) if templates_status and not self.use_raw_logs else 0,
                "opik_enabled": self.enable_opik_tracing
            }
            
        except Exception as e:
            logger.error(f"Error in health check: {e}")
            return {
                "success": False,
                "status": "error",
                "error": str(e)
            } 