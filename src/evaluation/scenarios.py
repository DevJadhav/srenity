"""Evaluation scenarios for SRE Workflow Agent."""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from loguru import logger

from .schemas import ScenarioType, GroundTruth
from ..models.schemas import IncidentRequest, SeverityLevel


class EvaluationScenarios:
    """Manages evaluation scenarios for SRE Workflow Agent."""
    
    def __init__(self):
        self.logger = logger.bind(component="EvaluationScenarios")
        self.scenarios = self._initialize_scenarios()
    
    def _initialize_scenarios(self) -> Dict[ScenarioType, Dict[str, Any]]:
        """Initialize all evaluation scenarios."""
        return {
            ScenarioType.SLOW_API_ENDPOINT: self._create_slow_api_endpoint_scenario(),
            ScenarioType.DISK_SPACE_ALERT: self._create_disk_space_scenario(),
            ScenarioType.HIGH_ERROR_RATE: self._create_high_error_rate_scenario(),
            ScenarioType.DATABASE_LOCK: self._create_database_lock_scenario(),
            ScenarioType.MEMORY_LEAK: self._create_memory_leak_scenario(),
            ScenarioType.NETWORK_CONNECTIVITY: self._create_network_connectivity_scenario()
        }
    
    def get_scenario(self, scenario_type: ScenarioType) -> Dict[str, Any]:
        """Get a specific scenario by type."""
        return self.scenarios.get(scenario_type, {})
    
    def get_all_scenarios(self) -> Dict[ScenarioType, Dict[str, Any]]:
        """Get all available scenarios."""
        return self.scenarios
    
    def create_custom_scenario(
        self,
        scenario_name: str,
        description: str,
        incident_request: IncidentRequest,
        ground_truth: GroundTruth,
        expected_behavior: str,
        mock_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a custom evaluation scenario."""
        return {
            "scenario_name": scenario_name,
            "description": description,
            "incident_request": incident_request,
            "ground_truth": ground_truth,
            "expected_behavior": expected_behavior,
            "mock_data": mock_data or {},
            "scenario_type": "custom"
        }
    
    def _create_slow_api_endpoint_scenario(self) -> Dict[str, Any]:
        """Create slow API endpoint scenario."""
        base_time = datetime.now() - timedelta(hours=1)
        
        incident_request = IncidentRequest(
            incident_id="INC-2024-001",
            title="Slow Customer-Facing API Endpoint",
            description="Synthetic monitoring detects /checkout endpoint latency > 5s",
            severity=SeverityLevel.HIGH,
            timestamp=base_time,
            affected_services=["checkout-service", "api-gateway", "payment-service"],
            metadata={
                "region": "us-east-1",
                "endpoint": "/checkout",
                "latency_threshold": "5s",
                "monitor_type": "synthetic"
            }
        )
        
        ground_truth = GroundTruth(
            scenario_id="slow_api_endpoint_001",
            incident_id="INC-2024-001",
            true_root_causes=[
                "Database connection pool exhaustion",
                "Slow database queries due to missing index"
            ],
            contributing_factors=[
                "Increased traffic during peak hours",
                "Recent code deployment without proper testing"
            ],
            expected_correlations=[
                {
                    "timestamp": base_time.isoformat(),
                    "events": ["API latency spike", "Database connection errors", "High CPU on DB server"],
                    "services": ["checkout-service", "postgres-db", "api-gateway"],
                    "correlation_type": "temporal"
                }
            ],
            event_timeline=[
                {"timestamp": (base_time - timedelta(minutes=30)).isoformat(), "event_type": "Code deployment", "service": "checkout-service"},
                {"timestamp": (base_time - timedelta(minutes=15)).isoformat(), "event_type": "Traffic increase", "service": "api-gateway"},
                {"timestamp": (base_time - timedelta(minutes=10)).isoformat(), "event_type": "Database connection errors", "service": "postgres-db"},
                {"timestamp": base_time.isoformat(), "event_type": "API latency alert", "service": "checkout-service"}
            ],
            expected_resolution_time_seconds=1800,  # 30 minutes
            expected_actions=[
                "Scale database connection pool",
                "Add missing database index",
                "Implement query optimization",
                "Set up connection pool monitoring"
            ],
            quality_score=0.85
        )
        
        mock_data = {
            "logs": [
                {
                    "timestamp": base_time.isoformat(),
                    "level": "error",
                    "service": "checkout-service",
                    "message": "Database connection timeout after 5000ms",
                    "correlation_id": "req-123"
                },
                {
                    "timestamp": (base_time + timedelta(minutes=2)).isoformat(),
                    "level": "warn",
                    "service": "postgres-db",
                    "message": "Connection pool exhausted: 95/100 connections in use",
                    "correlation_id": "db-pool-warn"
                }
            ],
            "metrics": [
                {
                    "metric_name": "api_latency_p95",
                    "value": 7.2,
                    "timestamp": base_time.isoformat(),
                    "service": "checkout-service",
                    "unit": "seconds"
                },
                {
                    "metric_name": "database_connections_active",
                    "value": 98,
                    "timestamp": base_time.isoformat(),
                    "service": "postgres-db",
                    "unit": "count"
                },
                {
                    "metric_name": "database_cpu_usage",
                    "value": 85.5,
                    "timestamp": base_time.isoformat(),
                    "service": "postgres-db",
                    "unit": "percent"
                }
            ],
            "traces": [
                {
                    "trace_id": "trace-123",
                    "span_id": "span-456",
                    "operation_name": "checkout_request",
                    "duration_ms": 7200,
                    "status": "error",
                    "service": "checkout-service",
                    "timestamp": base_time.isoformat()
                }
            ]
        }
        
        return {
            "scenario_name": "Slow Customer-Facing API Endpoint",
            "description": "Synthetic monitoring detects /checkout endpoint latency > 5s and created an incident",
            "incident_request": incident_request,
            "ground_truth": ground_truth,
            "expected_behavior": (
                "Should correlate with backend DB metrics, upstream service latencies. "
                "Highlight related code change or DB lock. "
                "Recommend profiling or increasing DB read replicas."
            ),
            "mock_data": mock_data,
            "scenario_type": ScenarioType.SLOW_API_ENDPOINT
        }
    
    def _create_disk_space_scenario(self) -> Dict[str, Any]:
        """Create disk space alert scenario."""
        base_time = datetime.now() - timedelta(hours=2)
        
        incident_request = IncidentRequest(
            incident_id="INC-2024-002",
            title="Disk Space Alert on Database Node",
            description="Disk usage > 90% on db-prod-3 and created an incident",
            severity=SeverityLevel.CRITICAL,
            timestamp=base_time,
            affected_services=["postgres-db", "backup-service"],
            metadata={
                "node": "db-prod-3",
                "disk_usage": "92%",
                "partition": "/var/lib/postgresql",
                "alert_threshold": "90%"
            }
        )
        
        ground_truth = GroundTruth(
            scenario_id="disk_space_alert_001",
            incident_id="INC-2024-002",
            true_root_causes=[
                "Log files not being rotated properly",
                "Backup files accumulating in data directory"
            ],
            contributing_factors=[
                "Backup retention policy not enforced",
                "Log rotation configuration missing"
            ],
            expected_correlations=[
                {
                    "timestamp": base_time.isoformat(),
                    "events": ["Disk space alert", "Backup job completion", "Log file growth"],
                    "services": ["postgres-db", "backup-service", "log-rotation"],
                    "correlation_type": "causal"
                }
            ],
            event_timeline=[
                {"timestamp": (base_time - timedelta(days=7)).isoformat(), "event_type": "Backup retention policy change", "service": "backup-service"},
                {"timestamp": (base_time - timedelta(days=3)).isoformat(), "event_type": "Log rotation service failure", "service": "log-rotation"},
                {"timestamp": (base_time - timedelta(hours=6)).isoformat(), "event_type": "Large backup file created", "service": "backup-service"},
                {"timestamp": base_time.isoformat(), "event_type": "Disk space alert triggered", "service": "postgres-db"}
            ],
            expected_resolution_time_seconds=3600,  # 1 hour
            expected_actions=[
                "Clean up old backup files",
                "Fix log rotation configuration",
                "Implement automated cleanup scripts",
                "Set up disk space monitoring with gradual alerts"
            ],
            quality_score=0.90
        )
        
        mock_data = {
            "logs": [
                {
                    "timestamp": base_time.isoformat(),
                    "level": "critical",
                    "service": "postgres-db",
                    "message": "Disk space critical: 92% used on /var/lib/postgresql",
                    "correlation_id": "disk-alert-001"
                },
                {
                    "timestamp": (base_time - timedelta(hours=1)).isoformat(),
                    "level": "error",
                    "service": "log-rotation",
                    "message": "Log rotation failed: Permission denied",
                    "correlation_id": "logrotate-error"
                }
            ],
            "metrics": [
                {
                    "metric_name": "disk_usage_percent",
                    "value": 92.0,
                    "timestamp": base_time.isoformat(),
                    "service": "postgres-db",
                    "unit": "percent"
                },
                {
                    "metric_name": "backup_files_count",
                    "value": 45,
                    "timestamp": base_time.isoformat(),
                    "service": "backup-service",
                    "unit": "count"
                }
            ],
            "traces": []
        }
        
        return {
            "scenario_name": "Disk Space Alert on Database Node",
            "description": "Disk usage > 90% on db-prod-3 and created an incident",
            "incident_request": incident_request,
            "ground_truth": ground_truth,
            "expected_behavior": (
                "Should show historical disk usage trends. "
                "Identify largest files/directories. "
                "Recommend archiving, auto-deletion, or volume expansion."
            ),
            "mock_data": mock_data,
            "scenario_type": ScenarioType.DISK_SPACE_ALERT
        }
    
    def _create_high_error_rate_scenario(self) -> Dict[str, Any]:
        """Create high error rate scenario."""
        base_time = datetime.now() - timedelta(minutes=30)
        
        incident_request = IncidentRequest(
            incident_id="INC-2024-003",
            title="High Error Rate in Web Service",
            description="Observing increased 500 errors in production web service",
            severity=SeverityLevel.HIGH,
            timestamp=base_time,
            affected_services=["web-service", "api-gateway", "user-service"],
            metadata={
                "error_rate": "15%",
                "baseline_error_rate": "0.5%",
                "error_types": ["500", "503"],
                "duration": "30 minutes"
            }
        )
        
        ground_truth = GroundTruth(
            scenario_id="high_error_rate_001",
            incident_id="INC-2024-003",
            true_root_causes=[
                "Downstream service timeout",
                "Circuit breaker not configured properly"
            ],
            contributing_factors=[
                "Increased load from marketing campaign",
                "Insufficient error handling in API calls"
            ],
            expected_correlations=[
                {
                    "timestamp": base_time.isoformat(),
                    "events": ["High error rate", "Downstream service latency", "Circuit breaker activation"],
                    "services": ["web-service", "user-service", "api-gateway"],
                    "correlation_type": "cascade"
                }
            ],
            event_timeline=[
                {"timestamp": (base_time - timedelta(hours=2)).isoformat(), "event_type": "Marketing campaign launch", "service": "external"},
                {"timestamp": (base_time - timedelta(minutes=45)).isoformat(), "event_type": "Traffic spike", "service": "api-gateway"},
                {"timestamp": (base_time - timedelta(minutes=35)).isoformat(), "event_type": "Downstream service slowdown", "service": "user-service"},
                {"timestamp": base_time.isoformat(), "event_type": "High error rate alert", "service": "web-service"}
            ],
            expected_resolution_time_seconds=2400,  # 40 minutes
            expected_actions=[
                "Configure circuit breaker with proper thresholds",
                "Implement retry logic with exponential backoff",
                "Scale downstream services",
                "Add proper error handling and fallback mechanisms"
            ],
            quality_score=0.80
        )
        
        mock_data = {
            "logs": [
                {
                    "timestamp": base_time.isoformat(),
                    "level": "error",
                    "service": "web-service",
                    "message": "HTTP 500 Internal Server Error: Timeout calling user-service",
                    "correlation_id": "req-456"
                },
                {
                    "timestamp": (base_time + timedelta(minutes=5)).isoformat(),
                    "level": "warn",
                    "service": "user-service",
                    "message": "High latency detected: 3.2s response time",
                    "correlation_id": "perf-warn"
                }
            ],
            "metrics": [
                {
                    "metric_name": "error_rate_percent",
                    "value": 15.0,
                    "timestamp": base_time.isoformat(),
                    "service": "web-service",
                    "unit": "percent"
                },
                {
                    "metric_name": "request_latency_p95",
                    "value": 3.2,
                    "timestamp": base_time.isoformat(),
                    "service": "user-service",
                    "unit": "seconds"
                }
            ],
            "traces": [
                {
                    "trace_id": "trace-789",
                    "span_id": "span-101",
                    "operation_name": "user_lookup",
                    "duration_ms": 5000,
                    "status": "timeout",
                    "service": "user-service",
                    "timestamp": base_time.isoformat()
                }
            ]
        }
        
        return {
            "scenario_name": "High Error Rate in Web Service",
            "description": "Observing increased 500 errors in production web service",
            "incident_request": incident_request,
            "ground_truth": ground_truth,
            "expected_behavior": (
                "Should identify cascade failure pattern. "
                "Correlate with downstream service performance. "
                "Recommend circuit breaker configuration and proper retry mechanisms."
            ),
            "mock_data": mock_data,
            "scenario_type": ScenarioType.HIGH_ERROR_RATE
        }
    
    def _create_database_lock_scenario(self) -> Dict[str, Any]:
        """Create database lock scenario."""
        base_time = datetime.now() - timedelta(minutes=45)
        
        incident_request = IncidentRequest(
            incident_id="INC-2024-004",
            title="Database Lock Contention",
            description="Database queries are timing out due to lock contention",
            severity=SeverityLevel.CRITICAL,
            timestamp=base_time,
            affected_services=["postgres-db", "order-service", "inventory-service"],
            metadata={
                "lock_type": "row_exclusive",
                "affected_table": "orders",
                "lock_duration": "45 minutes",
                "blocked_queries": 23
            }
        )
        
        ground_truth = GroundTruth(
            scenario_id="database_lock_001",
            incident_id="INC-2024-004",
            true_root_causes=[
                "Long-running transaction not committed",
                "Inefficient query without proper indexing"
            ],
            contributing_factors=[
                "Batch processing job running during peak hours",
                "Missing transaction timeout configuration"
            ],
            expected_correlations=[
                {
                    "timestamp": base_time.isoformat(),
                    "events": ["Database lock", "Query timeout", "Transaction rollback"],
                    "services": ["postgres-db", "order-service", "inventory-service"],
                    "correlation_type": "blocking"
                }
            ],
            event_timeline=[
                {"timestamp": (base_time - timedelta(minutes=60)).isoformat(), "event_type": "Batch job started", "service": "batch-processor"},
                {"timestamp": (base_time - timedelta(minutes=45)).isoformat(), "event_type": "Long transaction started", "service": "postgres-db"},
                {"timestamp": (base_time - timedelta(minutes=30)).isoformat(), "event_type": "Query timeouts begin", "service": "order-service"},
                {"timestamp": base_time.isoformat(), "event_type": "Database lock alert", "service": "postgres-db"}
            ],
            expected_resolution_time_seconds=1800,  # 30 minutes
            expected_actions=[
                "Kill long-running transaction",
                "Reschedule batch job to off-peak hours",
                "Add transaction timeout configuration",
                "Optimize query with proper indexing"
            ],
            quality_score=0.88
        )
        
        mock_data = {
            "logs": [
                {
                    "timestamp": base_time.isoformat(),
                    "level": "error",
                    "service": "postgres-db",
                    "message": "Lock timeout: transaction blocked for 45 minutes on table orders",
                    "correlation_id": "lock-timeout-001"
                },
                {
                    "timestamp": (base_time - timedelta(minutes=15)).isoformat(),
                    "level": "warn",
                    "service": "order-service",
                    "message": "Database query timeout: SELECT * FROM orders WHERE status = 'pending'",
                    "correlation_id": "query-timeout"
                }
            ],
            "metrics": [
                {
                    "metric_name": "database_locks_active",
                    "value": 23,
                    "timestamp": base_time.isoformat(),
                    "service": "postgres-db",
                    "unit": "count"
                },
                {
                    "metric_name": "query_duration_p95",
                    "value": 45.0,
                    "timestamp": base_time.isoformat(),
                    "service": "postgres-db",
                    "unit": "seconds"
                }
            ],
            "traces": [
                {
                    "trace_id": "trace-lock-001",
                    "span_id": "span-db-001",
                    "operation_name": "order_query",
                    "duration_ms": 45000,
                    "status": "timeout",
                    "service": "postgres-db",
                    "timestamp": base_time.isoformat()
                }
            ]
        }
        
        return {
            "scenario_name": "Database Lock Contention",
            "description": "Database queries are timing out due to lock contention",
            "incident_request": incident_request,
            "ground_truth": ground_truth,
            "expected_behavior": (
                "Should identify long-running transactions. "
                "Correlate with batch job scheduling. "
                "Recommend transaction timeout and query optimization."
            ),
            "mock_data": mock_data,
            "scenario_type": ScenarioType.DATABASE_LOCK
        }
    
    def _create_memory_leak_scenario(self) -> Dict[str, Any]:
        """Create memory leak scenario."""
        base_time = datetime.now() - timedelta(hours=4)
        
        incident_request = IncidentRequest(
            incident_id="INC-2024-005",
            title="Memory Leak in Application Service",
            description="Gradual memory increase leading to OOM kills",
            severity=SeverityLevel.HIGH,
            timestamp=base_time,
            affected_services=["app-service", "kubernetes"],
            metadata={
                "memory_usage": "95%",
                "oom_kills": 3,
                "pod_restarts": 5,
                "trend": "increasing"
            }
        )
        
        ground_truth = GroundTruth(
            scenario_id="memory_leak_001",
            incident_id="INC-2024-005",
            true_root_causes=[
                "Memory leak in cache implementation",
                "Unclosed database connections"
            ],
            contributing_factors=[
                "High request volume",
                "Inadequate garbage collection tuning"
            ],
            expected_correlations=[
                {
                    "timestamp": base_time.isoformat(),
                    "events": ["Memory usage spike", "OOM kill", "Pod restart"],
                    "services": ["app-service", "kubernetes"],
                    "correlation_type": "sequential"
                }
            ],
            event_timeline=[
                {"timestamp": (base_time - timedelta(hours=8)).isoformat(), "event_type": "Memory usage starts climbing", "service": "app-service"},
                {"timestamp": (base_time - timedelta(hours=4)).isoformat(), "event_type": "First OOM kill", "service": "kubernetes"},
                {"timestamp": (base_time - timedelta(hours=2)).isoformat(), "event_type": "Pod restart", "service": "kubernetes"},
                {"timestamp": base_time.isoformat(), "event_type": "Memory leak alert", "service": "app-service"}
            ],
            expected_resolution_time_seconds=7200,  # 2 hours
            expected_actions=[
                "Implement proper cache eviction policy",
                "Fix database connection leaks",
                "Tune garbage collection parameters",
                "Add memory monitoring and alerts"
            ],
            quality_score=0.82
        )
        
        mock_data = {
            "logs": [
                {
                    "timestamp": base_time.isoformat(),
                    "level": "error",
                    "service": "kubernetes",
                    "message": "OOM killed pod app-service-xyz due to memory limit exceeded",
                    "correlation_id": "oom-kill-001"
                },
                {
                    "timestamp": (base_time - timedelta(hours=1)).isoformat(),
                    "level": "warn",
                    "service": "app-service",
                    "message": "Memory usage high: 4.2GB / 4.5GB limit",
                    "correlation_id": "memory-warn"
                }
            ],
            "metrics": [
                {
                    "metric_name": "memory_usage_percent",
                    "value": 95.0,
                    "timestamp": base_time.isoformat(),
                    "service": "app-service",
                    "unit": "percent"
                },
                {
                    "metric_name": "gc_frequency",
                    "value": 45,
                    "timestamp": base_time.isoformat(),
                    "service": "app-service",
                    "unit": "per_minute"
                }
            ],
            "traces": []
        }
        
        return {
            "scenario_name": "Memory Leak in Application Service",
            "description": "Gradual memory increase leading to OOM kills",
            "incident_request": incident_request,
            "ground_truth": ground_truth,
            "expected_behavior": (
                "Should identify gradual memory increase pattern. "
                "Correlate with OOM kills and pod restarts. "
                "Recommend memory profiling and connection leak fixes."
            ),
            "mock_data": mock_data,
            "scenario_type": ScenarioType.MEMORY_LEAK
        }
    
    def _create_network_connectivity_scenario(self) -> Dict[str, Any]:
        """Create network connectivity scenario."""
        base_time = datetime.now() - timedelta(minutes=20)
        
        incident_request = IncidentRequest(
            incident_id="INC-2024-006",
            title="Network Connectivity Issues",
            description="Intermittent network connectivity causing service disruptions",
            severity=SeverityLevel.CRITICAL,
            timestamp=base_time,
            affected_services=["api-gateway", "auth-service", "payment-service"],
            metadata={
                "affected_region": "us-west-2",
                "packet_loss": "12%",
                "network_latency": "250ms",
                "connectivity_type": "intermittent"
            }
        )
        
        ground_truth = GroundTruth(
            scenario_id="network_connectivity_001",
            incident_id="INC-2024-006",
            true_root_causes=[
                "Network switch failure in data center",
                "Routing table corruption"
            ],
            contributing_factors=[
                "Lack of redundant network paths",
                "Insufficient network monitoring"
            ],
            expected_correlations=[
                {
                    "timestamp": base_time.isoformat(),
                    "events": ["Network timeout", "Connection refused", "Service unavailable"],
                    "services": ["api-gateway", "auth-service", "payment-service"],
                    "correlation_type": "infrastructure"
                }
            ],
            event_timeline=[
                {"timestamp": (base_time - timedelta(minutes=30)).isoformat(), "event_type": "Network switch degradation", "service": "infrastructure"},
                {"timestamp": (base_time - timedelta(minutes=25)).isoformat(), "event_type": "Packet loss increases", "service": "network"},
                {"timestamp": (base_time - timedelta(minutes=20)).isoformat(), "event_type": "Service timeouts begin", "service": "api-gateway"},
                {"timestamp": base_time.isoformat(), "event_type": "Network connectivity alert", "service": "monitoring"}
            ],
            expected_resolution_time_seconds=1200,  # 20 minutes
            expected_actions=[
                "Switch to backup network path",
                "Replace failed network switch",
                "Update routing tables",
                "Implement network redundancy"
            ],
            quality_score=0.85
        )
        
        mock_data = {
            "logs": [
                {
                    "timestamp": base_time.isoformat(),
                    "level": "error",
                    "service": "api-gateway",
                    "message": "Connection timeout to auth-service: No route to host",
                    "correlation_id": "network-timeout-001"
                },
                {
                    "timestamp": (base_time + timedelta(minutes=2)).isoformat(),
                    "level": "error",
                    "service": "payment-service",
                    "message": "Network error: Connection refused by upstream service",
                    "correlation_id": "network-error-002"
                }
            ],
            "metrics": [
                {
                    "metric_name": "packet_loss_percent",
                    "value": 12.0,
                    "timestamp": base_time.isoformat(),
                    "service": "network",
                    "unit": "percent"
                },
                {
                    "metric_name": "network_latency_ms",
                    "value": 250.0,
                    "timestamp": base_time.isoformat(),
                    "service": "network",
                    "unit": "milliseconds"
                }
            ],
            "traces": [
                {
                    "trace_id": "trace-network-001",
                    "span_id": "span-auth-001",
                    "operation_name": "auth_request",
                    "duration_ms": 30000,
                    "status": "timeout",
                    "service": "auth-service",
                    "timestamp": base_time.isoformat()
                }
            ]
        }
        
        return {
            "scenario_name": "Network Connectivity Issues",
            "description": "Intermittent network connectivity causing service disruptions",
            "incident_request": incident_request,
            "ground_truth": ground_truth,
            "expected_behavior": (
                "Should identify network infrastructure issues. "
                "Correlate service timeouts with network metrics. "
                "Recommend network redundancy and infrastructure fixes."
            ),
            "mock_data": mock_data,
            "scenario_type": ScenarioType.NETWORK_CONNECTIVITY
        }
    
 