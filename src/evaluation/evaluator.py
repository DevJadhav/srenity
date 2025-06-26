"""Main evaluator for SRE Workflow Agent."""

import asyncio
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from loguru import logger

from .metrics import (
    RCAEvaluationMetrics,
    CorrelationEvaluationMetrics,
    LLMEvaluationMetrics,
    SREOperationalMetrics,
    PerformanceMetrics
)
from .scenarios import EvaluationScenarios
from .schemas import (
    EvaluationResult,
    EvaluationConfig,
    ScenarioResult,
    EvaluationStatus,
    ScenarioType,
    RCAEvaluation,
    CorrelationEvaluation,
    LLMEvaluation,
    OperationalEvaluation,
    PerformanceEvaluation,
    GroundTruth
)
from ..agent.workflow import SREWorkflowAgent
from ..models.schemas import IncidentRequest, IncidentResponse


class SRenityEvaluator:
    """Main evaluator for SRE Workflow Agent."""
    
    def __init__(self, workflow_agent: SREWorkflowAgent):
        self.workflow_agent = workflow_agent
        self.logger = logger.bind(component="SRenityEvaluator")
        
        # Initialize evaluation components
        self.rca_metrics = RCAEvaluationMetrics()
        self.correlation_metrics = CorrelationEvaluationMetrics()
        self.llm_metrics = LLMEvaluationMetrics()
        self.operational_metrics = SREOperationalMetrics()
        self.performance_metrics = PerformanceMetrics()
        self.scenarios = EvaluationScenarios()
        
        self.logger.info("SRenity Evaluator initialized successfully")
    
    async def run_evaluation(self, config: EvaluationConfig) -> EvaluationResult:
        """Run complete evaluation based on configuration."""
        self.logger.info(f"Starting evaluation: {config.evaluation_name}")
        
        evaluation_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Create output directory
            output_dir = Path(config.output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize performance monitoring
            self.performance_metrics.start_monitoring()
            
            # Run scenario evaluations
            scenario_results = await self._run_scenarios(config)
            
            # Calculate aggregated metrics
            rca_evaluation = self._aggregate_rca_evaluations(scenario_results)
            correlation_evaluation = self._aggregate_correlation_evaluations(scenario_results)
            llm_evaluation = self._aggregate_llm_evaluations(scenario_results)
            operational_evaluation = self._aggregate_operational_evaluations(scenario_results)
            
            # Calculate performance metrics
            end_time = time.time()
            total_execution_time = end_time - start_time
            
            performance_evaluation = self.performance_metrics.evaluate_performance(
                analysis_time=total_execution_time,
                data_processing_time=sum(r.execution_time_seconds * 0.3 for r in scenario_results),
                llm_response_time=sum(r.execution_time_seconds * 0.5 for r in scenario_results),
                incidents_processed=len(scenario_results)
            )
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(
                rca_evaluation, correlation_evaluation, llm_evaluation,
                operational_evaluation, performance_evaluation
            )
            
            # Calculate scenario success rate
            passed_scenarios = sum(1 for r in scenario_results if r.success)
            scenario_success_rate = passed_scenarios / len(scenario_results) if scenario_results else 0.0
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                scenario_results, rca_evaluation, correlation_evaluation,
                llm_evaluation, operational_evaluation, performance_evaluation
            )
            
            # Create evaluation result
            evaluation_result = EvaluationResult(
                evaluation_id=evaluation_id,
                timestamp=datetime.now(),
                status=EvaluationStatus.COMPLETED,
                overall_score=overall_score,
                rca_evaluation=rca_evaluation,
                correlation_evaluation=correlation_evaluation,
                llm_evaluation=llm_evaluation,
                operational_evaluation=operational_evaluation,
                performance_evaluation=performance_evaluation,
                scenario_results=scenario_results,
                scenario_success_rate=scenario_success_rate,
                total_scenarios=len(scenario_results),
                passed_scenarios=passed_scenarios,
                failed_scenarios=len(scenario_results) - passed_scenarios,
                total_execution_time_seconds=total_execution_time,
                configuration=config.model_dump(),
                recommendations=recommendations
            )
            
            # Save results if configured
            if config.save_detailed_results:
                await self._save_evaluation_results(evaluation_result, output_dir)
            
            # Generate report if configured
            if config.generate_report:
                await self._generate_evaluation_report(evaluation_result, output_dir)
            
            self.logger.info(f"Evaluation completed successfully: {evaluation_id}")
            return evaluation_result
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            
            # Return failed evaluation result
            return EvaluationResult(
                evaluation_id=evaluation_id,
                timestamp=datetime.now(),
                status=EvaluationStatus.FAILED,
                overall_score=0.0,
                rca_evaluation=self._create_empty_rca_evaluation(),
                correlation_evaluation=self._create_empty_correlation_evaluation(),
                llm_evaluation=self._create_empty_llm_evaluation(),
                operational_evaluation=self._create_empty_operational_evaluation(),
                performance_evaluation=self._create_empty_performance_evaluation(),
                scenario_results=[],
                scenario_success_rate=0.0,
                total_scenarios=0,
                passed_scenarios=0,
                failed_scenarios=0,
                total_execution_time_seconds=time.time() - start_time,
                configuration=config.model_dump(),
                recommendations=[],
                error_summary=[str(e)]
            )
    
    async def _run_scenarios(self, config: EvaluationConfig) -> List[ScenarioResult]:
        """Run evaluation scenarios."""
        scenario_results = []
        
        # Get scenarios to run
        scenarios_to_evaluate = []
        
        # Add predefined scenarios
        for scenario_type in config.scenarios_to_run:
            scenario_data = self.scenarios.get_scenario(scenario_type)
            if scenario_data:
                scenarios_to_evaluate.append((scenario_type, scenario_data))
        
        # Add custom scenarios
        for custom_scenario in config.custom_scenarios:
            scenarios_to_evaluate.append(("custom", custom_scenario))
        
        # Run scenarios (parallel or sequential based on config)
        if config.parallel_execution:
            tasks = [
                self._run_single_scenario(scenario_type, scenario_data, config)
                for scenario_type, scenario_data in scenarios_to_evaluate
            ]
            scenario_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            for i, result in enumerate(scenario_results):
                if isinstance(result, Exception):
                    scenario_type, scenario_data = scenarios_to_evaluate[i]
                    scenario_results[i] = self._create_failed_scenario_result(
                        scenario_type, scenario_data, str(result)
                    )
        else:
            for scenario_type, scenario_data in scenarios_to_evaluate:
                result = await self._run_single_scenario(scenario_type, scenario_data, config)
                scenario_results.append(result)
        
        return scenario_results
    
    async def _run_single_scenario(
        self,
        scenario_type: ScenarioType,
        scenario_data: Dict[str, Any],
        config: EvaluationConfig
    ) -> ScenarioResult:
        """Run a single evaluation scenario."""
        scenario_id = f"{scenario_type}_{int(time.time())}"
        start_time = time.time()
        
        try:
            self.logger.info(f"Running scenario: {scenario_data.get('scenario_name', scenario_id)}")
            
            # Extract scenario components
            incident_request = scenario_data["incident_request"]
            ground_truth = scenario_data["ground_truth"]
            expected_behavior = scenario_data["expected_behavior"]
            mock_data = scenario_data.get("mock_data", {})
            
            # Run the workflow
            workflow_result = await self.workflow_agent.analyze_incident(incident_request)
            
            if not workflow_result.get("success"):
                raise Exception(f"Workflow failed: {workflow_result.get('error')}")
            
            incident_response_data = workflow_result["incident_response"]
            incident_response = IncidentResponse(**incident_response_data)
            
            # Evaluate different aspects
            rca_eval = self.rca_metrics.evaluate_rca(
                incident_response.root_cause_analysis, ground_truth
            )
            
            correlation_eval = self.correlation_metrics.evaluate_correlation(
                detected_correlations=self._extract_correlations_from_response(incident_response),
                ground_truth=ground_truth,
                processing_time_ms=(time.time() - start_time) * 1000
            )
            
            llm_eval = self.llm_metrics.evaluate_llm_response(
                incident_response, ground_truth, mock_data
            )
            
            # Calculate scenario score
            scenario_score = self._calculate_scenario_score(rca_eval, correlation_eval, llm_eval)
            
            # Determine success based on score threshold
            success = scenario_score >= 0.7  # 70% threshold
            
            execution_time = time.time() - start_time
            
            return ScenarioResult(
                scenario_id=scenario_id,
                scenario_type=scenario_type,
                scenario_name=scenario_data.get("scenario_name", str(scenario_type)),
                description=scenario_data.get("description", ""),
                expected_behavior=expected_behavior,
                actual_behavior=self._extract_actual_behavior(incident_response),
                success=success,
                score=scenario_score,
                execution_time_seconds=execution_time,
                detailed_results={
                    "rca_evaluation": rca_eval.model_dump(),
                    "correlation_evaluation": correlation_eval.model_dump(),
                    "llm_evaluation": llm_eval.model_dump(),
                    "incident_response": incident_response.model_dump(),
                    "workflow_result": workflow_result
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Scenario failed: {scenario_id} - {e}")
            
            return ScenarioResult(
                scenario_id=scenario_id,
                scenario_type=scenario_type,
                scenario_name=scenario_data.get("scenario_name", str(scenario_type)),
                description=scenario_data.get("description", ""),
                expected_behavior=expected_behavior,
                actual_behavior=f"Error: {str(e)}",
                success=False,
                score=0.0,
                execution_time_seconds=execution_time,
                error_message=str(e),
                detailed_results={"error": str(e)}
            )
    
    def _extract_correlations_from_response(self, response: IncidentResponse) -> List[Dict[str, Any]]:
        """Extract correlations from incident response."""
        correlations = []
        
        # Extract from timeline
        for event in response.root_cause_analysis.timeline:
            correlations.append({
                "timestamp": event.get("timestamp"),
                "events": [event.get("event", "")],
                "services": [event.get("service", "")]
            })
        
        return correlations
    
    def _extract_actual_behavior(self, response: IncidentResponse) -> str:
        """Extract actual behavior description from response."""
        behavior_parts = [
            f"RCA: {response.root_cause_analysis.primary_cause}",
            f"Recommendations: {len(response.recommendations)} actions provided",
            f"Next Steps: {len(response.next_steps)} steps outlined"
        ]
        
        return "; ".join(behavior_parts)
    
    def _calculate_scenario_score(
        self,
        rca_eval: RCAEvaluation,
        correlation_eval: CorrelationEvaluation,
        llm_eval: LLMEvaluation
    ) -> float:
        """Calculate overall scenario score."""
        weights = {
            "rca": 0.4,
            "correlation": 0.3,
            "llm": 0.3
        }
        
        rca_score = (rca_eval.precision + rca_eval.recall + rca_eval.f1_score) / 3
        correlation_score = correlation_eval.temporal_correlation_score
        llm_score = llm_eval.response_quality_score
        
        overall_score = (
            weights["rca"] * rca_score +
            weights["correlation"] * correlation_score +
            weights["llm"] * llm_score
        )
        
        return min(1.0, max(0.0, overall_score))
    
    def _aggregate_rca_evaluations(self, scenario_results: List[ScenarioResult]) -> RCAEvaluation:
        """Aggregate RCA evaluations across scenarios."""
        rca_evals = []
        
        for result in scenario_results:
            if result.success and "rca_evaluation" in result.detailed_results:
                rca_evals.append(RCAEvaluation(**result.detailed_results["rca_evaluation"]))
        
        if not rca_evals:
            return self._create_empty_rca_evaluation()
        
        # Calculate averages
        avg_precision = sum(e.precision for e in rca_evals) / len(rca_evals)
        avg_recall = sum(e.recall for e in rca_evals) / len(rca_evals)
        avg_f1 = sum(e.f1_score for e in rca_evals) / len(rca_evals)
        avg_accuracy = sum(e.accuracy for e in rca_evals) / len(rca_evals)
        
        # Aggregate other metrics
        all_predicted = []
        all_actual = []
        all_confidence = []
        
        for e in rca_evals:
            all_predicted.extend(e.predicted_causes)
            all_actual.extend(e.actual_causes)
            all_confidence.extend(e.confidence_scores)
        
        total_tp = sum(e.true_positives for e in rca_evals)
        total_fp = sum(e.false_positives for e in rca_evals)
        total_fn = sum(e.false_negatives for e in rca_evals)
        
        return RCAEvaluation(
            precision=avg_precision,
            recall=avg_recall,
            f1_score=avg_f1,
            accuracy=avg_accuracy,
            predicted_causes=list(set(all_predicted)),
            actual_causes=list(set(all_actual)),
            true_positives=total_tp,
            false_positives=total_fp,
            false_negatives=total_fn,
            confidence_scores=all_confidence
        )
    
    def _aggregate_correlation_evaluations(self, scenario_results: List[ScenarioResult]) -> CorrelationEvaluation:
        """Aggregate correlation evaluations across scenarios."""
        corr_evals = []
        
        for result in scenario_results:
            if result.success and "correlation_evaluation" in result.detailed_results:
                corr_evals.append(CorrelationEvaluation(**result.detailed_results["correlation_evaluation"]))
        
        if not corr_evals:
            return self._create_empty_correlation_evaluation()
        
        # Calculate averages
        avg_temporal_score = sum(e.temporal_correlation_score for e in corr_evals) / len(corr_evals)
        avg_precision = sum(e.correlation_precision for e in corr_evals) / len(corr_evals)
        avg_recall = sum(e.correlation_recall for e in corr_evals) / len(corr_evals)
        avg_ordering = sum(e.event_ordering_accuracy for e in corr_evals) / len(corr_evals)
        avg_cross_service = sum(e.cross_service_correlation for e in corr_evals) / len(corr_evals)
        avg_latency = sum(e.correlation_latency_ms for e in corr_evals) / len(corr_evals)
        
        # Aggregate correlations
        all_detected = []
        all_missed = []
        
        for e in corr_evals:
            all_detected.extend(e.detected_correlations)
            all_missed.extend(e.missed_correlations)
        
        return CorrelationEvaluation(
            temporal_correlation_score=avg_temporal_score,
            correlation_precision=avg_precision,
            correlation_recall=avg_recall,
            event_ordering_accuracy=avg_ordering,
            cross_service_correlation=avg_cross_service,
            correlation_latency_ms=avg_latency,
            detected_correlations=all_detected,
            missed_correlations=all_missed
        )
    
    def _aggregate_llm_evaluations(self, scenario_results: List[ScenarioResult]) -> LLMEvaluation:
        """Aggregate LLM evaluations across scenarios."""
        llm_evals = []
        
        for result in scenario_results:
            if result.success and "llm_evaluation" in result.detailed_results:
                llm_evals.append(LLMEvaluation(**result.detailed_results["llm_evaluation"]))
        
        if not llm_evals:
            return self._create_empty_llm_evaluation()
        
        # Calculate averages
        avg_hallucination_rate = sum(e.hallucination_rate for e in llm_evals) / len(llm_evals)
        avg_coherence = sum(e.coherence_score for e in llm_evals) / len(llm_evals)
        avg_grounding = sum(e.grounding_score for e in llm_evals) / len(llm_evals)
        avg_relevance = sum(e.relevance_score for e in llm_evals) / len(llm_evals)
        avg_completeness = sum(e.completeness_score for e in llm_evals) / len(llm_evals)
        avg_factual_accuracy = sum(e.factual_accuracy for e in llm_evals) / len(llm_evals)
        avg_quality = sum(e.response_quality_score for e in llm_evals) / len(llm_evals)
        
        # Aggregate claims and statements
        all_hallucinated = []
        all_ungrounded = []
        
        for e in llm_evals:
            all_hallucinated.extend(e.hallucinated_claims)
            all_ungrounded.extend(e.ungrounded_statements)
        
        return LLMEvaluation(
            hallucination_rate=avg_hallucination_rate,
            coherence_score=avg_coherence,
            grounding_score=avg_grounding,
            relevance_score=avg_relevance,
            completeness_score=avg_completeness,
            factual_accuracy=avg_factual_accuracy,
            hallucinated_claims=list(set(all_hallucinated)),
            ungrounded_statements=list(set(all_ungrounded)),
            response_quality_score=avg_quality
        )
    
    def _aggregate_operational_evaluations(self, scenario_results: List[ScenarioResult]) -> OperationalEvaluation:
        """Aggregate operational evaluations across scenarios."""
        # Create mock operational data based on scenario results
        incidents_data = []
        
        for result in scenario_results:
            incidents_data.append({
                "id": result.scenario_id,
                "start_time": datetime.now().isoformat(),
                "resolution_time": (datetime.now()).isoformat(),
                "duration_hours": result.execution_time_seconds / 3600,
                "severity": "high" if not result.success else "medium",
                "false_positive": not result.success and result.score < 0.3,
                "automated_response": result.success,
                "escalated": not result.success,
                "baseline_customer_impact": 100,
                "actual_customer_impact": 20 if result.success else 80
            })
        
        return self.operational_metrics.evaluate_operational_performance(incidents_data)
    
    def _calculate_overall_score(
        self,
        rca_eval: RCAEvaluation,
        correlation_eval: CorrelationEvaluation,
        llm_eval: LLMEvaluation,
        operational_eval: OperationalEvaluation,
        performance_eval: PerformanceEvaluation
    ) -> float:
        """Calculate overall evaluation score."""
        weights = {
            "rca": 0.25,
            "correlation": 0.20,
            "llm": 0.25,
            "operational": 0.20,
            "performance": 0.10
        }
        
        rca_score = (rca_eval.precision + rca_eval.recall + rca_eval.f1_score) / 3
        correlation_score = correlation_eval.temporal_correlation_score
        llm_score = llm_eval.response_quality_score
        operational_score = (
            operational_eval.incident_resolution_accuracy +
            operational_eval.cost_efficiency_score
        ) / 2
        performance_score = performance_eval.scalability_score
        
        overall_score = (
            weights["rca"] * rca_score +
            weights["correlation"] * correlation_score +
            weights["llm"] * llm_score +
            weights["operational"] * operational_score +
            weights["performance"] * performance_score
        )
        
        return min(1.0, max(0.0, overall_score))
    
    def _generate_recommendations(
        self,
        scenario_results: List[ScenarioResult],
        rca_eval: RCAEvaluation,
        correlation_eval: CorrelationEvaluation,
        llm_eval: LLMEvaluation,
        operational_eval: OperationalEvaluation,
        performance_eval: PerformanceEvaluation
    ) -> List[str]:
        """Generate improvement recommendations based on evaluation results."""
        recommendations = []
        
        # RCA recommendations
        if rca_eval.precision < 0.7:
            recommendations.append("Improve RCA precision by refining root cause identification logic")
        if rca_eval.recall < 0.7:
            recommendations.append("Enhance RCA recall by expanding root cause detection coverage")
        
        # Correlation recommendations
        if correlation_eval.temporal_correlation_score < 0.7:
            recommendations.append("Improve temporal correlation accuracy with better time-series analysis")
        if correlation_eval.event_ordering_accuracy < 0.8:
            recommendations.append("Enhance event ordering detection with improved timeline reconstruction")
        
        # LLM recommendations
        if llm_eval.hallucination_rate > 0.1:
            recommendations.append("Reduce hallucination rate by improving data grounding and fact-checking")
        if llm_eval.coherence_score < 0.8:
            recommendations.append("Improve response coherence with better logical flow structuring")
        
        # Operational recommendations
        if operational_eval.mttr_seconds > 3600:  # > 1 hour
            recommendations.append("Reduce MTTR by optimizing incident resolution workflows")
        if operational_eval.false_positive_rate > 0.2:
            recommendations.append("Reduce false positive rate by improving alert accuracy")
        
        # Performance recommendations
        if performance_eval.analysis_time_seconds > 300:  # > 5 minutes
            recommendations.append("Optimize analysis performance for faster incident resolution")
        if performance_eval.memory_usage_mb > 1000:  # > 1GB
            recommendations.append("Optimize memory usage for better resource efficiency")
        
        # Scenario-specific recommendations
        failed_scenarios = [r for r in scenario_results if not r.success]
        if failed_scenarios:
            scenario_types = list(set(r.scenario_type for r in failed_scenarios))
            recommendations.append(f"Address failures in scenario types: {', '.join(map(str, scenario_types))}")
        
        return recommendations
    
    def _create_failed_scenario_result(
        self,
        scenario_type: ScenarioType,
        scenario_data: Dict[str, Any],
        error_message: str
    ) -> ScenarioResult:
        """Create a failed scenario result."""
        return ScenarioResult(
            scenario_id=f"failed_{scenario_type}_{int(time.time())}",
            scenario_type=scenario_type,
            scenario_name=scenario_data.get("scenario_name", str(scenario_type)),
            description=scenario_data.get("description", ""),
            expected_behavior=scenario_data.get("expected_behavior", ""),
            actual_behavior=f"Failed: {error_message}",
            success=False,
            score=0.0,
            execution_time_seconds=0.0,
            error_message=error_message,
            detailed_results={"error": error_message}
        )
    
    def _create_empty_rca_evaluation(self) -> RCAEvaluation:
        """Create empty RCA evaluation for failed cases."""
        return RCAEvaluation(
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            accuracy=0.0,
            predicted_causes=[],
            actual_causes=[],
            true_positives=0,
            false_positives=0,
            false_negatives=0,
            confidence_scores=[]
        )
    
    def _create_empty_correlation_evaluation(self) -> CorrelationEvaluation:
        """Create empty correlation evaluation for failed cases."""
        return CorrelationEvaluation(
            temporal_correlation_score=0.0,
            correlation_precision=0.0,
            correlation_recall=0.0,
            event_ordering_accuracy=0.0,
            cross_service_correlation=0.0,
            correlation_latency_ms=0.0,
            detected_correlations=[],
            missed_correlations=[]
        )
    
    def _create_empty_llm_evaluation(self) -> LLMEvaluation:
        """Create empty LLM evaluation for failed cases."""
        return LLMEvaluation(
            hallucination_rate=1.0,
            coherence_score=0.0,
            grounding_score=0.0,
            relevance_score=0.0,
            completeness_score=0.0,
            factual_accuracy=0.0,
            hallucinated_claims=[],
            ungrounded_statements=[],
            response_quality_score=0.0
        )
    
    def _create_empty_operational_evaluation(self) -> OperationalEvaluation:
        """Create empty operational evaluation for failed cases."""
        return OperationalEvaluation(
            mttr_seconds=0.0,
            mttr_improvement_percent=0.0,
            incident_resolution_accuracy=0.0,
            cost_efficiency_score=0.0,
            false_positive_rate=1.0,
            automation_coverage=0.0,
            escalation_rate=1.0,
            customer_impact_reduction=0.0
        )
    
    def _create_empty_performance_evaluation(self) -> PerformanceEvaluation:
        """Create empty performance evaluation for failed cases."""
        return PerformanceEvaluation(
            analysis_time_seconds=0.0,
            data_processing_time_seconds=0.0,
            llm_response_time_seconds=0.0,
            memory_usage_mb=0.0,
            cpu_usage_percent=0.0,
            throughput_incidents_per_hour=0.0,
            api_response_time_ms=0.0,
            scalability_score=0.0
        )
    
    async def _save_evaluation_results(self, result: EvaluationResult, output_dir: Path):
        """Save evaluation results to file."""
        try:
            results_file = output_dir / f"evaluation_results_{result.evaluation_id}.json"
            
            with open(results_file, 'w') as f:
                f.write(result.model_dump_json(indent=2))
            
            self.logger.info(f"Evaluation results saved to: {results_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
    
    async def _generate_evaluation_report(self, result: EvaluationResult, output_dir: Path):
        """Generate human-readable evaluation report."""
        try:
            report_file = output_dir / f"evaluation_report_{result.evaluation_id}.md"
            
            report_content = self._create_markdown_report(result)
            
            with open(report_file, 'w') as f:
                f.write(report_content)
            
            self.logger.info(f"Evaluation report generated: {report_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate report: {e}")
    
    def _create_markdown_report(self, result: EvaluationResult) -> str:
        """Create markdown evaluation report."""
        report = f"""# SRenity Evaluation Report

## Overview
- **Evaluation ID**: {result.evaluation_id}
- **Timestamp**: {result.timestamp}
- **Status**: {result.status}
- **Overall Score**: {result.overall_score:.3f}

## Summary
- **Total Scenarios**: {result.total_scenarios}
- **Passed Scenarios**: {result.passed_scenarios}
- **Failed Scenarios**: {result.failed_scenarios}
- **Success Rate**: {result.scenario_success_rate:.1%}
- **Execution Time**: {result.total_execution_time_seconds:.2f} seconds

## Detailed Metrics

### Root Cause Analysis (RCA)
- **Precision**: {result.rca_evaluation.precision:.3f}
- **Recall**: {result.rca_evaluation.recall:.3f}
- **F1 Score**: {result.rca_evaluation.f1_score:.3f}
- **Accuracy**: {result.rca_evaluation.accuracy:.3f}

### Event Correlation
- **Temporal Correlation Score**: {result.correlation_evaluation.temporal_correlation_score:.3f}
- **Correlation Precision**: {result.correlation_evaluation.correlation_precision:.3f}
- **Correlation Recall**: {result.correlation_evaluation.correlation_recall:.3f}
- **Event Ordering Accuracy**: {result.correlation_evaluation.event_ordering_accuracy:.3f}

### LLM Quality
- **Hallucination Rate**: {result.llm_evaluation.hallucination_rate:.3f}
- **Coherence Score**: {result.llm_evaluation.coherence_score:.3f}
- **Grounding Score**: {result.llm_evaluation.grounding_score:.3f}
- **Response Quality**: {result.llm_evaluation.response_quality_score:.3f}

### Operational Metrics
- **MTTR**: {result.operational_evaluation.mttr_seconds:.1f} seconds
- **Resolution Accuracy**: {result.operational_evaluation.incident_resolution_accuracy:.3f}
- **Cost Efficiency**: {result.operational_evaluation.cost_efficiency_score:.3f}
- **False Positive Rate**: {result.operational_evaluation.false_positive_rate:.3f}

### Performance Metrics
- **Analysis Time**: {result.performance_evaluation.analysis_time_seconds:.2f} seconds
- **Memory Usage**: {result.performance_evaluation.memory_usage_mb:.1f} MB
- **CPU Usage**: {result.performance_evaluation.cpu_usage_percent:.1f}%
- **Throughput**: {result.performance_evaluation.throughput_incidents_per_hour:.1f} incidents/hour

## Scenario Results
"""
        
        for scenario in result.scenario_results:
            status = "✅ PASSED" if scenario.success else "❌ FAILED"
            report += f"""
### {scenario.scenario_name} {status}
- **Score**: {scenario.score:.3f}
- **Execution Time**: {scenario.execution_time_seconds:.2f} seconds
- **Expected**: {scenario.expected_behavior}
- **Actual**: {scenario.actual_behavior}
"""
            if scenario.error_message:
                report += f"- **Error**: {scenario.error_message}\n"
        
        report += f"""
## Recommendations
"""
        for i, rec in enumerate(result.recommendations, 1):
            report += f"{i}. {rec}\n"
        
        if result.error_summary:
            report += f"""
## Errors Encountered
"""
            for error in result.error_summary:
                report += f"- {error}\n"
        
        return report 