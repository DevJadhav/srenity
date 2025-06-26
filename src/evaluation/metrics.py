"""Evaluation metrics for SRE Workflow Agent."""

import time
import psutil
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from loguru import logger

from .schemas import (
    RCAEvaluation,
    CorrelationEvaluation,
    LLMEvaluation,
    OperationalEvaluation,
    PerformanceEvaluation,
    GroundTruth
)
from ..models.schemas import (
    RootCauseAnalysis,
    IncidentResponse,
    LogEntry,
    MetricData,
    TraceData
)


class RCAEvaluationMetrics:
    """Root Cause Analysis evaluation metrics calculator."""
    
    def __init__(self):
        self.logger = logger.bind(component="RCAEvaluationMetrics")
    
    def evaluate_rca(
        self,
        predicted_rca: RootCauseAnalysis,
        ground_truth: GroundTruth
    ) -> RCAEvaluation:
        """Evaluate Root Cause Analysis predictions."""
        self.logger.info("Evaluating RCA predictions")
        
        # Extract predicted and actual causes
        predicted_causes = [predicted_rca.primary_cause] + predicted_rca.contributing_factors
        actual_causes = ground_truth.true_root_causes + ground_truth.contributing_factors
        
        # Calculate metrics
        precision, recall, f1, accuracy, tp, fp, fn = self._calculate_classification_metrics(
            predicted_causes, actual_causes
        )
        
        # Extract confidence scores
        confidence_scores = [predicted_rca.confidence_score] * len(predicted_causes)
        
        return RCAEvaluation(
            precision=precision,
            recall=recall,
            f1_score=f1,
            accuracy=accuracy,
            predicted_causes=predicted_causes,
            actual_causes=actual_causes,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            confidence_scores=confidence_scores
        )
    
    def _calculate_classification_metrics(
        self,
        predicted: List[str],
        actual: List[str]
    ) -> Tuple[float, float, float, float, int, int, int]:
        """Calculate precision, recall, F1, accuracy for string-based predictions."""
        
        # Convert to sets for comparison
        predicted_set = set(cause.lower().strip() for cause in predicted)
        actual_set = set(cause.lower().strip() for cause in actual)
        
        # Calculate set-based metrics
        tp = len(predicted_set.intersection(actual_set))
        fp = len(predicted_set - actual_set)
        fn = len(actual_set - predicted_set)
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = tp / len(actual_set.union(predicted_set)) if len(actual_set.union(predicted_set)) > 0 else 0.0
        
        return precision, recall, f1, accuracy, tp, fp, fn


class CorrelationEvaluationMetrics:
    """Event correlation evaluation metrics calculator."""
    
    def __init__(self):
        self.logger = logger.bind(component="CorrelationEvaluationMetrics")
    
    def evaluate_correlation(
        self,
        detected_correlations: List[Dict[str, Any]],
        ground_truth: GroundTruth,
        processing_time_ms: float
    ) -> CorrelationEvaluation:
        """Evaluate event correlation performance."""
        self.logger.info("Evaluating event correlation")
        
        expected_correlations = ground_truth.expected_correlations
        
        # Calculate temporal correlation score
        temporal_score = self._calculate_temporal_correlation_score(
            detected_correlations, expected_correlations
        )
        
        # Calculate precision and recall for correlations
        correlation_precision, correlation_recall = self._calculate_correlation_precision_recall(
            detected_correlations, expected_correlations
        )
        
        # Calculate event ordering accuracy
        ordering_accuracy = self._calculate_event_ordering_accuracy(
            detected_correlations, ground_truth.event_timeline
        )
        
        # Calculate cross-service correlation accuracy
        cross_service_accuracy = self._calculate_cross_service_correlation_accuracy(
            detected_correlations, expected_correlations
        )
        
        # Identify missed correlations
        missed_correlations = self._identify_missed_correlations(
            detected_correlations, expected_correlations
        )
        
        return CorrelationEvaluation(
            temporal_correlation_score=temporal_score,
            correlation_precision=correlation_precision,
            correlation_recall=correlation_recall,
            event_ordering_accuracy=ordering_accuracy,
            cross_service_correlation=cross_service_accuracy,
            correlation_latency_ms=processing_time_ms,
            detected_correlations=detected_correlations,
            missed_correlations=missed_correlations
        )
    
    def _calculate_temporal_correlation_score(
        self,
        detected: List[Dict[str, Any]],
        expected: List[Dict[str, Any]]
    ) -> float:
        """Calculate temporal correlation accuracy score."""
        if not expected:
            return 1.0 if not detected else 0.0
        
        score = 0.0
        for exp_corr in expected:
            best_match_score = 0.0
            for det_corr in detected:
                # Calculate temporal similarity
                time_similarity = self._calculate_time_similarity(
                    det_corr.get('timestamp'), exp_corr.get('timestamp')
                )
                # Calculate event similarity
                event_similarity = self._calculate_event_similarity(
                    det_corr.get('events', []), exp_corr.get('events', [])
                )
                
                match_score = 0.7 * time_similarity + 0.3 * event_similarity
                best_match_score = max(best_match_score, match_score)
            
            score += best_match_score
        
        return score / len(expected)
    
    def _calculate_time_similarity(self, time1: Optional[str], time2: Optional[str]) -> float:
        """Calculate similarity between two timestamps."""
        if not time1 or not time2:
            return 0.0
        
        try:
            dt1 = datetime.fromisoformat(time1.replace('Z', '+00:00'))
            dt2 = datetime.fromisoformat(time2.replace('Z', '+00:00'))
            
            # Calculate time difference in seconds
            diff_seconds = abs((dt1 - dt2).total_seconds())
            
            # Convert to similarity score (closer to 0 seconds = higher score)
            max_acceptable_diff = 300  # 5 minutes
            similarity = max(0.0, 1.0 - (diff_seconds / max_acceptable_diff))
            
            return min(1.0, similarity)
        except Exception:
            return 0.0
    
    def _calculate_event_similarity(self, events1: List[str], events2: List[str]) -> float:
        """Calculate similarity between two event lists."""
        if not events1 and not events2:
            return 1.0
        if not events1 or not events2:
            return 0.0
        
        set1 = set(event.lower().strip() for event in events1)
        set2 = set(event.lower().strip() for event in events2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_correlation_precision_recall(
        self,
        detected: List[Dict[str, Any]],
        expected: List[Dict[str, Any]]
    ) -> Tuple[float, float]:
        """Calculate precision and recall for correlations."""
        if not expected:
            return 1.0 if not detected else 0.0, 1.0
        
        if not detected:
            return 0.0, 0.0
        
        # Count true positives (correlations that match expected ones)
        true_positives = 0
        for exp_corr in expected:
            for det_corr in detected:
                if self._correlations_match(det_corr, exp_corr):
                    true_positives += 1
                    break
        
        precision = true_positives / len(detected)
        recall = true_positives / len(expected)
        
        return precision, recall
    
    def _correlations_match(self, detected: Dict[str, Any], expected: Dict[str, Any]) -> bool:
        """Check if two correlations match."""
        # Simple matching based on event similarity and time proximity
        event_similarity = self._calculate_event_similarity(
            detected.get('events', []), expected.get('events', [])
        )
        time_similarity = self._calculate_time_similarity(
            detected.get('timestamp'), expected.get('timestamp')
        )
        
        return event_similarity > 0.7 and time_similarity > 0.7
    
    def _calculate_event_ordering_accuracy(
        self,
        detected_correlations: List[Dict[str, Any]],
        expected_timeline: List[Dict[str, Any]]
    ) -> float:
        """Calculate accuracy of event ordering."""
        if not expected_timeline:
            return 1.0
        
        # Extract event ordering from detected correlations
        detected_ordering = self._extract_event_ordering(detected_correlations)
        expected_ordering = [event.get('event_type') for event in expected_timeline]
        
        if not detected_ordering:
            return 0.0
        
        # Calculate ordering similarity using longest common subsequence
        lcs_length = self._longest_common_subsequence(detected_ordering, expected_ordering)
        
        return lcs_length / len(expected_ordering) if expected_ordering else 1.0
    
    def _extract_event_ordering(self, correlations: List[Dict[str, Any]]) -> List[str]:
        """Extract event ordering from correlations."""
        events_with_time = []
        
        for corr in correlations:
            timestamp = corr.get('timestamp')
            events = corr.get('events', [])
            
            for event in events:
                events_with_time.append((timestamp, event))
        
        # Sort by timestamp and extract events
        events_with_time.sort(key=lambda x: x[0] if x[0] else '')
        
        return [event for _, event in events_with_time]
    
    def _longest_common_subsequence(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate length of longest common subsequence."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1].lower() == seq2[j-1].lower():
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def _calculate_cross_service_correlation_accuracy(
        self,
        detected: List[Dict[str, Any]],
        expected: List[Dict[str, Any]]
    ) -> float:
        """Calculate cross-service correlation accuracy."""
        # Filter to cross-service correlations only
        detected_cross_service = [
            corr for corr in detected
            if len(set(corr.get('services', []))) > 1
        ]
        
        expected_cross_service = [
            corr for corr in expected
            if len(set(corr.get('services', []))) > 1
        ]
        
        if not expected_cross_service:
            return 1.0 if not detected_cross_service else 0.0
        
        # Calculate precision and recall for cross-service correlations
        precision, recall = self._calculate_correlation_precision_recall(
            detected_cross_service, expected_cross_service
        )
        
        # Return F1 score as overall cross-service accuracy
        if precision + recall > 0:
            return 2 * (precision * recall) / (precision + recall)
        else:
            return 0.0
    
    def _identify_missed_correlations(
        self,
        detected: List[Dict[str, Any]],
        expected: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Identify correlations that were missed."""
        missed = []
        
        for exp_corr in expected:
            found_match = False
            for det_corr in detected:
                if self._correlations_match(det_corr, exp_corr):
                    found_match = True
                    break
            
            if not found_match:
                missed.append(exp_corr)
        
        return missed


class LLMEvaluationMetrics:
    """LLM-specific evaluation metrics calculator."""
    
    def __init__(self, evaluation_model: str = "gpt-4"):
        self.evaluation_model = evaluation_model
        self.logger = logger.bind(component="LLMEvaluationMetrics")
    
    def evaluate_llm_response(
        self,
        incident_response: IncidentResponse,
        ground_truth: GroundTruth,
        supporting_data: Dict[str, Any]
    ) -> LLMEvaluation:
        """Evaluate LLM response quality."""
        self.logger.info("Evaluating LLM response quality")
        
        # Extract response text for analysis
        response_text = self._extract_response_text(incident_response)
        
        # Calculate hallucination rate
        hallucination_rate, hallucinated_claims = self._calculate_hallucination_rate(
            response_text, supporting_data
        )
        
        # Calculate coherence score
        coherence_score = self._calculate_coherence_score(response_text)
        
        # Calculate grounding score
        grounding_score, ungrounded_statements = self._calculate_grounding_score(
            response_text, supporting_data
        )
        
        # Calculate relevance score
        relevance_score = self._calculate_relevance_score(
            response_text, incident_response.incident_id
        )
        
        # Calculate completeness score
        completeness_score = self._calculate_completeness_score(
            incident_response, ground_truth
        )
        
        # Calculate factual accuracy
        factual_accuracy = self._calculate_factual_accuracy(
            incident_response, ground_truth
        )
        
        # Calculate overall response quality
        response_quality = self._calculate_overall_quality(
            coherence_score, grounding_score, relevance_score, 
            completeness_score, factual_accuracy, hallucination_rate
        )
        
        return LLMEvaluation(
            hallucination_rate=hallucination_rate,
            coherence_score=coherence_score,
            grounding_score=grounding_score,
            relevance_score=relevance_score,
            completeness_score=completeness_score,
            factual_accuracy=factual_accuracy,
            hallucinated_claims=hallucinated_claims,
            ungrounded_statements=ungrounded_statements,
            response_quality_score=response_quality
        )
    
    def _extract_response_text(self, response: IncidentResponse) -> str:
        """Extract all text content from incident response."""
        text_parts = [
            response.summary,
            response.root_cause_analysis.primary_cause,
            " ".join(response.root_cause_analysis.contributing_factors),
            " ".join([rec.action for rec in response.recommendations]),
            " ".join(response.next_steps)
        ]
        
        return " ".join(filter(None, text_parts))
    
    def _calculate_hallucination_rate(
        self,
        response_text: str,
        supporting_data: Dict[str, Any]
    ) -> Tuple[float, List[str]]:
        """Calculate hallucination rate - claims not supported by data."""
        # This would ideally use an LLM to evaluate hallucinations
        # For now, using heuristic-based detection
        
        hallucinated_claims = []
        
        # Extract claims that mention specific metrics, services, or times
        claims = self._extract_specific_claims(response_text)
        
        # Check each claim against supporting data
        for claim in claims:
            if not self._is_claim_supported(claim, supporting_data):
                hallucinated_claims.append(claim)
        
        hallucination_rate = len(hallucinated_claims) / len(claims) if claims else 0.0
        
        return hallucination_rate, hallucinated_claims
    
    def _extract_specific_claims(self, text: str) -> List[str]:
        """Extract specific claims that can be fact-checked."""
        # Simple heuristic - split by sentences and filter for specific claims
        sentences = text.split('.')
        
        specific_claims = []
        keywords = ['error rate', 'latency', 'cpu', 'memory', 'disk', 'response time', '%', 'seconds', 'minutes']
        
        for sentence in sentences:
            sentence = sentence.strip()
            if any(keyword in sentence.lower() for keyword in keywords):
                specific_claims.append(sentence)
        
        return specific_claims
    
    def _is_claim_supported(self, claim: str, supporting_data: Dict[str, Any]) -> bool:
        """Check if a claim is supported by the available data."""
        # Simplified check - in practice, this would be more sophisticated
        claim_lower = claim.lower()
        
        # Check if claim mentions data that doesn't exist in supporting data
        if 'error rate' in claim_lower and 'error_metrics' not in supporting_data:
            return False
        
        if any(term in claim_lower for term in ['cpu', 'memory', 'disk']) and 'system_metrics' not in supporting_data:
            return False
        
        # More checks would be added here
        return True
    
    def _calculate_coherence_score(self, response_text: str) -> float:
        """Calculate logical flow coherence score."""
        # Simple heuristic-based coherence scoring
        sentences = response_text.split('.')
        
        if len(sentences) < 2:
            return 1.0
        
        # Check for logical flow indicators
        flow_indicators = ['therefore', 'consequently', 'as a result', 'because', 'due to', 'since']
        coherence_signals = sum(1 for sentence in sentences if any(indicator in sentence.lower() for indicator in flow_indicators))
        
        # Check for contradiction indicators
        contradiction_indicators = ['however', 'but', 'although', 'despite']
        contradictions = sum(1 for sentence in sentences if any(indicator in sentence.lower() for indicator in contradiction_indicators))
        
        # Simple scoring formula
        coherence_ratio = coherence_signals / len(sentences)
        contradiction_penalty = contradictions / len(sentences)
        
        score = min(1.0, coherence_ratio + 0.5) - contradiction_penalty * 0.3
        
        return max(0.0, score)
    
    def _calculate_grounding_score(
        self,
        response_text: str,
        supporting_data: Dict[str, Any]
    ) -> Tuple[float, List[str]]:
        """Calculate how well the response is grounded in supporting data."""
        sentences = response_text.split('.')
        ungrounded_statements = []
        grounded_count = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if self._is_statement_grounded(sentence, supporting_data):
                grounded_count += 1
            else:
                ungrounded_statements.append(sentence)
        
        grounding_score = grounded_count / len([s for s in sentences if s.strip()]) if sentences else 1.0
        
        return grounding_score, ungrounded_statements
    
    def _is_statement_grounded(self, statement: str, supporting_data: Dict[str, Any]) -> bool:
        """Check if a statement is grounded in supporting data."""
        # Simplified grounding check
        statement_lower = statement.lower()
        
        # Check for log data references
        if 'log' in statement_lower or 'error' in statement_lower:
            return 'logs' in supporting_data and len(supporting_data.get('logs', [])) > 0
        
        # Check for metric references
        if any(term in statement_lower for term in ['metric', 'cpu', 'memory', 'latency']):
            return 'metrics' in supporting_data and len(supporting_data.get('metrics', [])) > 0
        
        # Check for trace references
        if 'trace' in statement_lower or 'request' in statement_lower:
            return 'traces' in supporting_data and len(supporting_data.get('traces', [])) > 0
        
        # If no specific data type mentioned, consider it grounded
        return True
    
    def _calculate_relevance_score(self, response_text: str, incident_id: str) -> float:
        """Calculate relevance of response to the incident."""
        # Simple relevance scoring based on incident-related keywords
        incident_keywords = ['incident', 'error', 'failure', 'issue', 'problem', 'outage']
        solution_keywords = ['fix', 'resolve', 'solution', 'recommendation', 'action']
        
        text_lower = response_text.lower()
        
        incident_mentions = sum(1 for keyword in incident_keywords if keyword in text_lower)
        solution_mentions = sum(1 for keyword in solution_keywords if keyword in text_lower)
        
        # Calculate relevance as combination of incident understanding and solution provision
        relevance_score = min(1.0, (incident_mentions + solution_mentions) / 10)
        
        return max(0.3, relevance_score)  # Minimum baseline relevance
    
    def _calculate_completeness_score(
        self,
        response: IncidentResponse,
        ground_truth: GroundTruth
    ) -> float:
        """Calculate completeness of the response."""
        completeness_factors = []
        
        # Check if RCA is provided
        completeness_factors.append(1.0 if response.root_cause_analysis.primary_cause else 0.0)
        
        # Check if recommendations are provided
        completeness_factors.append(1.0 if response.recommendations else 0.0)
        
        # Check if next steps are provided
        completeness_factors.append(1.0 if response.next_steps else 0.0)
        
        # Check if summary is provided
        completeness_factors.append(1.0 if response.summary else 0.0)
        
        # Check if timeline is provided
        completeness_factors.append(1.0 if response.root_cause_analysis.timeline else 0.0)
        
        return sum(completeness_factors) / len(completeness_factors)
    
    def _calculate_factual_accuracy(
        self,
        response: IncidentResponse,
        ground_truth: GroundTruth
    ) -> float:
        """Calculate factual accuracy against ground truth."""
        if not ground_truth.true_root_causes:
            return 1.0  # No ground truth to compare against
        
        # Compare root causes
        predicted_causes = [response.root_cause_analysis.primary_cause] + response.root_cause_analysis.contributing_factors
        actual_causes = ground_truth.true_root_causes + ground_truth.contributing_factors
        
        # Calculate overlap
        predicted_set = set(cause.lower().strip() for cause in predicted_causes)
        actual_set = set(cause.lower().strip() for cause in actual_causes)
        
        if not actual_set:
            return 1.0
        
        overlap = len(predicted_set.intersection(actual_set))
        accuracy = overlap / len(actual_set)
        
        return accuracy
    
    def _calculate_overall_quality(
        self,
        coherence: float,
        grounding: float,
        relevance: float,
        completeness: float,
        factual_accuracy: float,
        hallucination_rate: float
    ) -> float:
        """Calculate overall response quality score."""
        # Weighted combination of quality factors
        weights = {
            'coherence': 0.15,
            'grounding': 0.25,
            'relevance': 0.15,
            'completeness': 0.15,
            'factual_accuracy': 0.25,
            'hallucination_penalty': 0.05
        }
        
        quality_score = (
            weights['coherence'] * coherence +
            weights['grounding'] * grounding +
            weights['relevance'] * relevance +
            weights['completeness'] * completeness +
            weights['factual_accuracy'] * factual_accuracy -
            weights['hallucination_penalty'] * hallucination_rate
        )
        
        return max(0.0, min(1.0, quality_score))


class SREOperationalMetrics:
    """SRE/Operational evaluation metrics calculator."""
    
    def __init__(self):
        self.logger = logger.bind(component="SREOperationalMetrics")
    
    def evaluate_operational_performance(
        self,
        incidents_data: List[Dict[str, Any]],
        baseline_mttr: Optional[float] = None
    ) -> OperationalEvaluation:
        """Evaluate operational performance metrics."""
        self.logger.info("Evaluating operational performance")
        
        if not incidents_data:
            # Return default values if no data
            return OperationalEvaluation(
                mttr_seconds=0.0,
                mttr_improvement_percent=0.0,
                incident_resolution_accuracy=0.0,
                cost_efficiency_score=0.0,
                false_positive_rate=0.0,
                automation_coverage=0.0,
                escalation_rate=0.0,
                customer_impact_reduction=0.0
            )
        
        # Calculate MTTR
        mttr = self._calculate_mttr(incidents_data)
        
        # Calculate MTTR improvement
        mttr_improvement = self._calculate_mttr_improvement(mttr, baseline_mttr)
        
        # Calculate incident resolution accuracy
        resolution_accuracy = self._calculate_resolution_accuracy(incidents_data)
        
        # Calculate cost efficiency
        cost_efficiency = self._calculate_cost_efficiency(incidents_data)
        
        # Calculate false positive rate
        false_positive_rate = self._calculate_false_positive_rate(incidents_data)
        
        # Calculate automation coverage
        automation_coverage = self._calculate_automation_coverage(incidents_data)
        
        # Calculate escalation rate
        escalation_rate = self._calculate_escalation_rate(incidents_data)
        
        # Calculate customer impact reduction
        impact_reduction = self._calculate_customer_impact_reduction(incidents_data)
        
        return OperationalEvaluation(
            mttr_seconds=mttr,
            mttr_improvement_percent=mttr_improvement,
            incident_resolution_accuracy=resolution_accuracy,
            cost_efficiency_score=cost_efficiency,
            false_positive_rate=false_positive_rate,
            automation_coverage=automation_coverage,
            escalation_rate=escalation_rate,
            customer_impact_reduction=impact_reduction
        )
    
    def _calculate_mttr(self, incidents_data: List[Dict[str, Any]]) -> float:
        """Calculate Mean Time to Resolution."""
        resolution_times = []
        
        for incident in incidents_data:
            start_time = incident.get('start_time')
            resolution_time = incident.get('resolution_time')
            
            if start_time and resolution_time:
                try:
                    start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                    resolution_dt = datetime.fromisoformat(resolution_time.replace('Z', '+00:00'))
                    
                    resolution_duration = (resolution_dt - start_dt).total_seconds()
                    resolution_times.append(resolution_duration)
                except Exception as e:
                    self.logger.warning(f"Failed to parse time for incident {incident.get('id')}: {e}")
        
        return sum(resolution_times) / len(resolution_times) if resolution_times else 0.0
    
    def _calculate_mttr_improvement(self, current_mttr: float, baseline_mttr: Optional[float]) -> float:
        """Calculate MTTR improvement percentage."""
        if not baseline_mttr or baseline_mttr == 0:
            return 0.0
        
        improvement = ((baseline_mttr - current_mttr) / baseline_mttr) * 100
        return improvement
    
    def _calculate_resolution_accuracy(self, incidents_data: List[Dict[str, Any]]) -> float:
        """Calculate accuracy of incident resolution."""
        correctly_resolved = 0
        total_incidents = len(incidents_data)
        
        for incident in incidents_data:
            # Check if incident was properly resolved (no recurrence within 24 hours)
            resolution_time = incident.get('resolution_time')
            recurrence_time = incident.get('recurrence_time')
            
            if resolution_time:
                if not recurrence_time:
                    correctly_resolved += 1
                else:
                    try:
                        res_dt = datetime.fromisoformat(resolution_time.replace('Z', '+00:00'))
                        rec_dt = datetime.fromisoformat(recurrence_time.replace('Z', '+00:00'))
                        
                        # If recurrence is more than 24 hours later, consider it correctly resolved
                        if (rec_dt - res_dt).total_seconds() > 24 * 3600:
                            correctly_resolved += 1
                    except Exception:
                        pass
        
        return correctly_resolved / total_incidents if total_incidents > 0 else 0.0
    
    def _calculate_cost_efficiency(self, incidents_data: List[Dict[str, Any]]) -> float:
        """Calculate cost efficiency score."""
        # Simplified cost efficiency calculation
        total_cost = 0.0
        total_incidents = len(incidents_data)
        
        for incident in incidents_data:
            # Estimate cost based on severity and duration
            severity = incident.get('severity', 'medium')
            duration_hours = incident.get('duration_hours', 1.0)
            
            # Cost multipliers by severity
            severity_multipliers = {
                'critical': 1000,
                'high': 500,
                'medium': 200,
                'low': 50
            }
            
            incident_cost = severity_multipliers.get(severity, 200) * duration_hours
            total_cost += incident_cost
        
        # Calculate efficiency as inverse of average cost per incident
        avg_cost = total_cost / total_incidents if total_incidents > 0 else 0
        
        # Normalize to 0-1 scale (assuming $10,000 as high cost threshold)
        efficiency_score = max(0.0, 1.0 - (avg_cost / 10000))
        
        return efficiency_score
    
    def _calculate_false_positive_rate(self, incidents_data: List[Dict[str, Any]]) -> float:
        """Calculate false positive alert rate."""
        false_positives = 0
        total_alerts = len(incidents_data)
        
        for incident in incidents_data:
            # Check if incident was marked as false positive
            if incident.get('false_positive', False):
                false_positives += 1
        
        return false_positives / total_alerts if total_alerts > 0 else 0.0
    
    def _calculate_automation_coverage(self, incidents_data: List[Dict[str, Any]]) -> float:
        """Calculate percentage of incidents handled automatically."""
        automated_responses = 0
        total_incidents = len(incidents_data)
        
        for incident in incidents_data:
            if incident.get('automated_response', False):
                automated_responses += 1
        
        return automated_responses / total_incidents if total_incidents > 0 else 0.0
    
    def _calculate_escalation_rate(self, incidents_data: List[Dict[str, Any]]) -> float:
        """Calculate rate of incident escalations."""
        escalated_incidents = 0
        total_incidents = len(incidents_data)
        
        for incident in incidents_data:
            if incident.get('escalated', False):
                escalated_incidents += 1
        
        return escalated_incidents / total_incidents if total_incidents > 0 else 0.0
    
    def _calculate_customer_impact_reduction(self, incidents_data: List[Dict[str, Any]]) -> float:
        """Calculate reduction in customer impact."""
        total_impact_reduction = 0.0
        incidents_with_impact_data = 0
        
        for incident in incidents_data:
            baseline_impact = incident.get('baseline_customer_impact', 0)
            actual_impact = incident.get('actual_customer_impact', 0)
            
            if baseline_impact > 0:
                impact_reduction = (baseline_impact - actual_impact) / baseline_impact
                total_impact_reduction += max(0.0, impact_reduction)
                incidents_with_impact_data += 1
        
        return total_impact_reduction / incidents_with_impact_data if incidents_with_impact_data > 0 else 0.0


class PerformanceMetrics:
    """Performance metrics calculator."""
    
    def __init__(self):
        self.logger = logger.bind(component="PerformanceMetrics")
        self._start_time = None
        self._start_memory = None
        self._start_cpu = None
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self._start_time = time.time()
        self._start_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
        self._start_cpu = psutil.cpu_percent()
    
    def evaluate_performance(
        self,
        analysis_time: float,
        data_processing_time: float,
        llm_response_time: float,
        incidents_processed: int = 1
    ) -> PerformanceEvaluation:
        """Evaluate performance metrics."""
        self.logger.info("Evaluating performance metrics")
        
        # Get current resource usage
        current_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
        current_cpu = psutil.cpu_percent()
        
        # Calculate metrics
        memory_usage = current_memory - (self._start_memory or current_memory)
        cpu_usage = (current_cpu + (self._start_cpu or 0)) / 2
        
        # Calculate throughput
        total_time_hours = analysis_time / 3600
        throughput = incidents_processed / total_time_hours if total_time_hours > 0 else 0
        
        # Calculate scalability score (simplified)
        scalability_score = self._calculate_scalability_score(
            analysis_time, memory_usage, cpu_usage
        )
        
        # API response time (mock for now)
        api_response_time = analysis_time * 1000  # Convert to ms
        
        return PerformanceEvaluation(
            analysis_time_seconds=analysis_time,
            data_processing_time_seconds=data_processing_time,
            llm_response_time_seconds=llm_response_time,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            throughput_incidents_per_hour=throughput,
            api_response_time_ms=api_response_time,
            scalability_score=scalability_score
        )
    
    def _calculate_scalability_score(
        self,
        analysis_time: float,
        memory_usage: float,
        cpu_usage: float
    ) -> float:
        """Calculate scalability score based on resource efficiency."""
        # Normalize metrics to 0-1 scale
        time_score = max(0.0, 1.0 - (analysis_time / 300))  # 5 minutes as threshold
        memory_score = max(0.0, 1.0 - (memory_usage / 1000))  # 1GB as threshold
        cpu_score = max(0.0, 1.0 - (cpu_usage / 100))
        
        # Weighted average
        scalability_score = 0.4 * time_score + 0.3 * memory_score + 0.3 * cpu_score
        
        return scalability_score 