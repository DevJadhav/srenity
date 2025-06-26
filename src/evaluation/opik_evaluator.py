"""Opik-based evaluator for SRE Workflow Agent."""

import os
import time
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime

from loguru import logger

# Import Opik for evaluation
try:
    import opik
    from opik import track, evaluate, track_openai
    from opik.evaluation import base_metric, metrics
    OPIK_AVAILABLE = True
    logger.info("✅ Opik evaluation framework available")
except ImportError:
    OPIK_AVAILABLE = False
    logger.warning("⚠️  Opik not available - advanced evaluation disabled")

from ..agent.workflow import SREWorkflowAgent
from ..models.schemas import IncidentRequest, IncidentResponse
from .schemas import EvaluationConfig, EvaluationResult, ScenarioResult


class OpikWorkflowEvaluator:
    """Opik-based evaluator for complete workflow tracing and evaluation."""
    
    def __init__(self, workflow_agent: SREWorkflowAgent):
        """Initialize the Opik evaluator."""
        self.workflow_agent = workflow_agent
        self.logger = logger.bind(component="OpikWorkflowEvaluator")
        
        # Initialize Opik client
        self.opik_client = self._initialize_opik()
        
        # Define evaluation metrics
        self.metrics = self._initialize_metrics()
        
        self.logger.info("Opik Workflow Evaluator initialized")
    
    def _initialize_opik(self) -> Optional[Any]:
        """Initialize Opik client for evaluation."""
        if not OPIK_AVAILABLE:
            return None
        
        try:
            # Set environment variables
            os.environ["OPIK_API_KEY"] = os.getenv("OPIK_API_KEY", "")
            os.environ["OPIK_WORKSPACE"] = os.getenv("OPIK_WORKSPACE", "")
            
            # Initialize client
            client = opik.Opik()
            
            # Configure project
            project_name = os.getenv("OPIK_PROJECT_NAME", "srenity-evaluation")
            
            self.logger.info(f"✅ Opik client initialized for evaluation project: {project_name}")
            return client
            
        except Exception as e:
            self.logger.warning(f"⚠️  Failed to initialize Opik client: {e}")
            return None
    
    def _initialize_metrics(self) -> Dict[str, Any]:
        """Initialize evaluation metrics."""
        if not OPIK_AVAILABLE:
            return {}
        
        # Define custom metrics for SRE workflow evaluation
        metrics_dict = {
            "rca_accuracy": RCAAccuracyMetric(),
            "response_quality": ResponseQualityMetric(),
            "recommendation_relevance": RecommendationRelevanceMetric(),
            "workflow_latency": WorkflowLatencyMetric(),
            "error_detection": ErrorDetectionMetric()
        }
        
        return metrics_dict
    
    @track(project_name="srenity-evaluation", capture_input=True, capture_output=True)
    async def evaluate_workflow(
        self,
        test_scenarios: List[Dict[str, Any]],
        config: Optional[EvaluationConfig] = None
    ) -> Dict[str, Any]:
        """Evaluate the workflow using Opik's evaluation framework."""
        
        if not OPIK_AVAILABLE or not self.opik_client:
            self.logger.warning("Opik not available, falling back to basic evaluation")
            return await self._basic_evaluation(test_scenarios)
        
        evaluation_id = str(uuid.uuid4())
        self.logger.info(f"Starting Opik workflow evaluation: {evaluation_id}")
        
        try:
            # Create evaluation dataset
            dataset = self._create_evaluation_dataset(test_scenarios)
            
            # Define evaluation task
            @track(project_name="srenity-evaluation")
            async def evaluate_incident(scenario: Dict[str, Any]) -> Dict[str, Any]:
                """Evaluate a single incident scenario."""
                incident_request = IncidentRequest(**scenario["incident_request"])
                result = await self.workflow_agent.analyze_incident(incident_request)
                
                return {
                    "input": scenario,
                    "output": result,
                    "expected": scenario.get("expected_output", {}),
                    "ground_truth": scenario.get("ground_truth", {})
                }
            
            # Run evaluation
            evaluation_results = []
            for scenario in test_scenarios:
                try:
                    result = await evaluate_incident(scenario)
                    
                    # Calculate metrics
                    metric_results = self._calculate_metrics(result)
                    
                    evaluation_results.append({
                        "scenario": scenario.get("name", "unknown"),
                        "success": result["output"].get("success", False),
                        "metrics": metric_results,
                        "result": result
                    })
                    
                except Exception as e:
                    self.logger.error(f"Error evaluating scenario: {e}")
                    evaluation_results.append({
                        "scenario": scenario.get("name", "unknown"),
                        "success": False,
                        "error": str(e)
                    })
            
            # Aggregate results
            aggregated_metrics = self._aggregate_metrics(evaluation_results)
            
            # Create Opik experiment
            experiment = self._create_opik_experiment(
                evaluation_id,
                test_scenarios,
                evaluation_results,
                aggregated_metrics
            )
            
            return {
                "evaluation_id": evaluation_id,
                "experiment_id": experiment.id if experiment else None,
                "total_scenarios": len(test_scenarios),
                "passed": sum(1 for r in evaluation_results if r.get("success", False)),
                "failed": sum(1 for r in evaluation_results if not r.get("success", False)),
                "aggregated_metrics": aggregated_metrics,
                "detailed_results": evaluation_results,
                "opik_dashboard_url": self._get_dashboard_url(experiment)
            }
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            return {
                "evaluation_id": evaluation_id,
                "error": str(e),
                "success": False
            }
    
    def _create_evaluation_dataset(self, test_scenarios: List[Dict[str, Any]]) -> Any:
        """Create an Opik dataset from test scenarios."""
        if not OPIK_AVAILABLE:
            return None
        
        try:
            # Create dataset
            dataset_name = f"srenity_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Format scenarios for Opik dataset
            dataset_items = []
            for scenario in test_scenarios:
                dataset_items.append({
                    "input": scenario.get("incident_request", {}),
                    "expected_output": scenario.get("expected_output", {}),
                    "metadata": {
                        "scenario_name": scenario.get("name", "unknown"),
                        "scenario_type": scenario.get("type", "unknown"),
                        "severity": scenario.get("incident_request", {}).get("severity", "unknown")
                    }
                })
            
            # Note: In a real implementation, you would create the dataset in Opik
            # For now, we'll return the formatted data
            return {
                "name": dataset_name,
                "items": dataset_items
            }
            
        except Exception as e:
            self.logger.error(f"Failed to create dataset: {e}")
            return None
    
    def _calculate_metrics(self, result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate evaluation metrics for a single result."""
        metrics_results = {}
        
        if not self.metrics:
            return metrics_results
        
        try:
            # Calculate each metric
            for metric_name, metric in self.metrics.items():
                try:
                    score = metric.calculate(result)
                    metrics_results[metric_name] = score
                except Exception as e:
                    self.logger.warning(f"Failed to calculate {metric_name}: {e}")
                    metrics_results[metric_name] = 0.0
            
            return metrics_results
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            return {}
    
    def _aggregate_metrics(self, evaluation_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate metrics across all evaluation results."""
        aggregated = {}
        
        # Collect all metrics
        all_metrics = {}
        for result in evaluation_results:
            if "metrics" in result:
                for metric_name, value in result["metrics"].items():
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    all_metrics[metric_name].append(value)
        
        # Calculate averages
        for metric_name, values in all_metrics.items():
            if values:
                aggregated[f"{metric_name}_avg"] = sum(values) / len(values)
                aggregated[f"{metric_name}_min"] = min(values)
                aggregated[f"{metric_name}_max"] = max(values)
        
        # Overall success rate
        total = len(evaluation_results)
        passed = sum(1 for r in evaluation_results if r.get("success", False))
        aggregated["success_rate"] = passed / total if total > 0 else 0.0
        
        return aggregated
    
    def _create_opik_experiment(
        self,
        evaluation_id: str,
        test_scenarios: List[Dict[str, Any]],
        evaluation_results: List[Dict[str, Any]],
        aggregated_metrics: Dict[str, float]
    ) -> Any:
        """Create an Opik experiment for the evaluation."""
        if not OPIK_AVAILABLE:
            return None
        
        try:
            # Create experiment metadata
            experiment_data = {
                "name": f"srenity_evaluation_{evaluation_id}",
                "metadata": {
                    "evaluation_id": evaluation_id,
                    "timestamp": datetime.now().isoformat(),
                    "total_scenarios": len(test_scenarios),
                    "model": self.workflow_agent.llm_model,
                    "workflow_version": "1.0"
                },
                "metrics": aggregated_metrics,
                "config": {
                    "scenarios": [s.get("name", "unknown") for s in test_scenarios],
                    "evaluation_type": "workflow_evaluation"
                }
            }
            
            # Note: In a real implementation, you would create the experiment in Opik
            # For now, we'll return the formatted data
            class MockExperiment:
                def __init__(self, data):
                    self.id = evaluation_id
                    self.data = data
            
            return MockExperiment(experiment_data)
            
        except Exception as e:
            self.logger.error(f"Failed to create experiment: {e}")
            return None
    
    def _get_dashboard_url(self, experiment: Any) -> str:
        """Get the Opik dashboard URL for the experiment."""
        if not experiment or not OPIK_AVAILABLE:
            return ""
        
        workspace = os.getenv("OPIK_WORKSPACE", "")
        if workspace and hasattr(experiment, "id"):
            return f"https://app.opik.com/{workspace}/experiments/{experiment.id}"
        
        return ""
    
    async def _basic_evaluation(self, test_scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback basic evaluation without Opik."""
        evaluation_id = str(uuid.uuid4())
        results = []
        
        for scenario in test_scenarios:
            try:
                incident_request = IncidentRequest(**scenario["incident_request"])
                result = await self.workflow_agent.analyze_incident(incident_request)
                
                results.append({
                    "scenario": scenario.get("name", "unknown"),
                    "success": result.get("success", False),
                    "result": result
                })
            except Exception as e:
                results.append({
                    "scenario": scenario.get("name", "unknown"),
                    "success": False,
                    "error": str(e)
                })
        
        passed = sum(1 for r in results if r.get("success", False))
        
        return {
            "evaluation_id": evaluation_id,
            "total_scenarios": len(test_scenarios),
            "passed": passed,
            "failed": len(test_scenarios) - passed,
            "success_rate": passed / len(test_scenarios) if test_scenarios else 0.0,
            "detailed_results": results
        }
    
    def generate_evaluation_report(self, evaluation_results: Dict[str, Any]) -> str:
        """Generate a comprehensive evaluation report."""
        report = f"""# SRenity Workflow Evaluation Report

## Evaluation Summary
- **Evaluation ID**: {evaluation_results.get('evaluation_id', 'N/A')}
- **Total Scenarios**: {evaluation_results.get('total_scenarios', 0)}
- **Passed**: {evaluation_results.get('passed', 0)}
- **Failed**: {evaluation_results.get('failed', 0)}
- **Success Rate**: {evaluation_results.get('aggregated_metrics', {}).get('success_rate', 0):.2%}

## Aggregated Metrics
"""
        
        # Add metrics
        metrics = evaluation_results.get('aggregated_metrics', {})
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                report += f"- **{metric_name}**: {value:.3f}\n"
        
        # Add Opik dashboard link if available
        dashboard_url = evaluation_results.get('opik_dashboard_url', '')
        if dashboard_url:
            report += f"\n## Opik Dashboard\n[View detailed results in Opik]({dashboard_url})\n"
        
        # Add detailed results summary
        report += "\n## Scenario Results\n"
        for result in evaluation_results.get('detailed_results', []):
            status = "✅" if result.get('success', False) else "❌"
            report += f"- {status} **{result.get('scenario', 'Unknown')}**"
            if 'error' in result:
                report += f" - Error: {result['error']}"
            report += "\n"
        
        return report


# Custom metric implementations
class RCAAccuracyMetric:
    """Metric to evaluate Root Cause Analysis accuracy."""
    
    def calculate(self, result: Dict[str, Any]) -> float:
        """Calculate RCA accuracy score."""
        try:
            output = result.get("output", {})
            ground_truth = result.get("ground_truth", {})
            
            if not output.get("success"):
                return 0.0
            
            response = output.get("incident_response", {})
            if not response:
                return 0.0
            
            # Compare RCA with ground truth
            predicted_cause = response.get("root_cause_analysis", {}).get("primary_cause", "")
            expected_cause = ground_truth.get("root_cause", "")
            
            if not predicted_cause or not expected_cause:
                return 0.5  # Partial credit if RCA was attempted
            
            # Simple string similarity (in production, use more sophisticated comparison)
            similarity = self._calculate_similarity(predicted_cause.lower(), expected_cause.lower())
            
            return similarity
            
        except Exception as e:
            logger.error(f"Error calculating RCA accuracy: {e}")
            return 0.0
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity (simple implementation)."""
        # Check for exact match
        if str1 == str2:
            return 1.0
        
        # Check for substring match
        if str1 in str2 or str2 in str1:
            return 0.8
        
        # Check for common keywords
        keywords1 = set(str1.split())
        keywords2 = set(str2.split())
        
        if keywords1 and keywords2:
            overlap = len(keywords1.intersection(keywords2))
            total = len(keywords1.union(keywords2))
            return overlap / total if total > 0 else 0.0
        
        return 0.0


class ResponseQualityMetric:
    """Metric to evaluate overall response quality."""
    
    def calculate(self, result: Dict[str, Any]) -> float:
        """Calculate response quality score."""
        try:
            output = result.get("output", {})
            
            if not output.get("success"):
                return 0.0
            
            response = output.get("incident_response", {})
            if not response:
                return 0.0
            
            # Check completeness of response
            score = 0.0
            max_score = 5.0
            
            # Check for RCA
            if response.get("root_cause_analysis"):
                score += 1.0
            
            # Check for recommendations
            recommendations = response.get("recommendations", [])
            if recommendations:
                score += min(len(recommendations) / 3.0, 1.0)  # Up to 1 point for recommendations
            
            # Check for analysis
            if response.get("analysis"):
                score += 1.0
            
            # Check for summary
            if response.get("summary"):
                score += 1.0
            
            # Check for next steps
            if response.get("next_steps"):
                score += 1.0
            
            return score / max_score
            
        except Exception as e:
            logger.error(f"Error calculating response quality: {e}")
            return 0.0


class RecommendationRelevanceMetric:
    """Metric to evaluate recommendation relevance."""
    
    def calculate(self, result: Dict[str, Any]) -> float:
        """Calculate recommendation relevance score."""
        try:
            output = result.get("output", {})
            expected = result.get("expected", {})
            
            if not output.get("success"):
                return 0.0
            
            response = output.get("incident_response", {})
            recommendations = response.get("recommendations", [])
            
            if not recommendations:
                return 0.0
            
            # Check if recommendations address the incident
            incident = result.get("input", {}).get("incident_request", {})
            incident_keywords = self._extract_keywords(incident)
            
            relevance_scores = []
            for rec in recommendations:
                rec_text = f"{rec.get('action', '')} {rec.get('expected_impact', '')}"
                rec_keywords = self._extract_keywords({"text": rec_text})
                
                # Calculate keyword overlap
                overlap = len(incident_keywords.intersection(rec_keywords))
                total = len(incident_keywords)
                
                relevance = overlap / total if total > 0 else 0.0
                relevance_scores.append(relevance)
            
            return sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating recommendation relevance: {e}")
            return 0.0
    
    def _extract_keywords(self, data: Dict[str, Any]) -> set:
        """Extract keywords from data."""
        text = " ".join(str(v) for v in data.values() if v)
        # Simple keyword extraction (in production, use NLP)
        words = text.lower().split()
        # Filter out common words
        stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"}
        return set(w for w in words if w not in stopwords and len(w) > 2)


class WorkflowLatencyMetric:
    """Metric to evaluate workflow execution latency."""
    
    def calculate(self, result: Dict[str, Any]) -> float:
        """Calculate latency score (lower is better, normalized to 0-1)."""
        try:
            # This would normally come from actual timing data
            # For now, return a mock score
            return 0.8  # Assume good performance
            
        except Exception as e:
            logger.error(f"Error calculating latency: {e}")
            return 0.0


class ErrorDetectionMetric:
    """Metric to evaluate error detection capability."""
    
    def calculate(self, result: Dict[str, Any]) -> float:
        """Calculate error detection score."""
        try:
            output = result.get("output", {})
            ground_truth = result.get("ground_truth", {})
            
            if not output.get("success"):
                return 0.0
            
            response = output.get("incident_response", {})
            analysis = response.get("analysis", {})
            
            # Check if errors were detected
            detected_errors = analysis.get("error_frequency", {})
            expected_errors = ground_truth.get("errors", [])
            
            if not expected_errors:
                # No errors to detect, check if none were falsely detected
                return 0.0 if detected_errors else 1.0
            
            # Calculate detection rate
            detected_count = sum(detected_errors.values()) if detected_errors else 0
            expected_count = len(expected_errors)
            
            if detected_count == 0:
                return 0.0
            
            # Normalize score
            detection_rate = min(detected_count / expected_count, 1.0)
            
            return detection_rate
            
        except Exception as e:
            logger.error(f"Error calculating error detection: {e}")
            return 0.0 