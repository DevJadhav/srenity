"""Data processors for logs, metrics, and traces."""

import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
from loguru import logger

from ..models.schemas import LogEntry, LogLevel, MetricData, TraceData


class LogProcessor:
    """Process Apache log data from CSV files."""
    
    def __init__(self, log_file_path: Union[str, Path], templates_file_path: Union[str, Path]):
        """Initialize the log processor with data file paths."""
        self.log_file_path = Path(log_file_path)
        self.templates_file_path = Path(templates_file_path)
        self.log_data: Optional[pd.DataFrame] = None
        self.templates_data: Optional[pd.DataFrame] = None
        self._load_data()
    
    def _load_data(self) -> None:
        """Load log and template data from CSV files."""
        try:
            # Load main log data
            self.log_data = pd.read_csv(self.log_file_path)
            logger.info(f"Loaded {len(self.log_data)} log entries from {self.log_file_path}")
            
            # Load templates data
            self.templates_data = pd.read_csv(self.templates_file_path)
            logger.info(f"Loaded {len(self.templates_data)} templates from {self.templates_file_path}")
            
            # Convert timestamps
            self.log_data['Time'] = pd.to_datetime(self.log_data['Time'], format='%a %b %d %H:%M:%S %Y')
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def get_logs_by_timeframe(
        self, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[LogEntry]:
        """Get logs within a specific timeframe."""
        if self.log_data is None:
            return []
        
        # Convert datetime objects to be timezone-naive to match pandas datetime
        if start_time.tzinfo is not None:
            start_time = start_time.replace(tzinfo=None)
        if end_time.tzinfo is not None:
            end_time = end_time.replace(tzinfo=None)
        
        # Filter logs by timeframe
        mask = (self.log_data['Time'] >= start_time) & (self.log_data['Time'] <= end_time)
        filtered_logs = self.log_data.loc[mask]
        
        # Convert to LogEntry objects
        log_entries = []
        for _, row in filtered_logs.iterrows():
            try:
                log_entry = LogEntry(
                    line_id=int(row['LineId']),
                    timestamp=row['Time'],
                    level=LogLevel(row['Level'].lower()),
                    content=str(row['Content']),
                    event_id=str(row['EventId']),
                    event_template=str(row['EventTemplate'])
                )
                log_entries.append(log_entry)
            except Exception as e:
                logger.warning(f"Error processing log entry {row['LineId']}: {e}")
                continue
        
        return log_entries
    
    def get_error_logs(self, timeframe_hours: int = 24) -> List[LogEntry]:
        """Get all error-level logs within the specified timeframe."""
        if self.log_data is None:
            return []
        
        # Calculate timeframe
        end_time = self.log_data['Time'].max()
        start_time = end_time - timedelta(hours=timeframe_hours)
        
        # Filter for error logs
        error_mask = (
            (self.log_data['Level'] == 'error') & 
            (self.log_data['Time'] >= start_time) & 
            (self.log_data['Time'] <= end_time)
        )
        error_logs = self.log_data.loc[error_mask]
        
        # Convert to LogEntry objects
        log_entries = []
        for _, row in error_logs.iterrows():
            try:
                log_entry = LogEntry(
                    line_id=int(row['LineId']),
                    timestamp=row['Time'],
                    level=LogLevel.ERROR,
                    content=str(row['Content']),
                    event_id=str(row['EventId']),
                    event_template=str(row['EventTemplate'])
                )
                log_entries.append(log_entry)
            except Exception as e:
                logger.warning(f"Error processing error log {row['LineId']}: {e}")
                continue
        
        return log_entries
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """Get general log statistics."""
        if self.log_data is None:
            return {}
        
        # Level distribution
        level_counts = self.log_data['Level'].value_counts().to_dict()
        
        # Event ID distribution
        event_counts = self.log_data['EventId'].value_counts().to_dict()
        
        # Time range
        time_range = {
            "start": self.log_data['Time'].min().isoformat(),
            "end": self.log_data['Time'].max().isoformat(),
            "duration_hours": (self.log_data['Time'].max() - self.log_data['Time'].min()).total_seconds() / 3600
        }
        
        return {
            "total_entries": len(self.log_data),
            "level_distribution": level_counts,
            "event_distribution": event_counts,
            "time_range": time_range
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Alias for get_log_statistics for compatibility."""
        stats = self.get_log_statistics()
        if stats:
            # Calculate error counts for compatibility
            error_count = stats.get("level_distribution", {}).get("error", 0)
            stats.update({
                "total_errors": error_count,
                "timespan_hours": stats.get("time_range", {}).get("duration_hours", 0)
            })
        return stats
    
    def analyze_error_patterns(self, timeframe_hours: int = 24) -> Dict[str, Any]:
        """Analyze error patterns in the logs."""
        if self.log_data is None:
            return {}
        
        # Get error logs
        error_logs = self.get_error_logs(timeframe_hours)
        
        if not error_logs:
            return {"total_errors": 0, "patterns": {}, "frequency": {}}
        
        # Count error patterns by event template
        pattern_counts = {}
        event_id_counts = {}
        hourly_distribution = {}
        
        for log in error_logs:
            # Count by event template
            if log.event_template not in pattern_counts:
                pattern_counts[log.event_template] = 0
            pattern_counts[log.event_template] += 1
            
            # Count by event ID
            if log.event_id not in event_id_counts:
                event_id_counts[log.event_id] = 0
            event_id_counts[log.event_id] += 1
            
            # Count by hour
            hour_key = log.timestamp.strftime('%Y-%m-%d %H:00')
            if hour_key not in hourly_distribution:
                hourly_distribution[hour_key] = 0
            hourly_distribution[hour_key] += 1
        
        return {
            "total_errors": len(error_logs),
            "patterns": pattern_counts,
            "event_frequency": event_id_counts,
            "hourly_distribution": hourly_distribution,
            "time_window": f"{timeframe_hours} hours"
        }


class MetricsProcessor:
    """Generate and process mock metrics data."""
    
    def __init__(self):
        """Initialize the metrics processor."""
        self.base_metrics = [
            "cpu_usage_percent",
            "memory_usage_percent",
            "disk_usage_percent",
            "network_bytes_in",
            "network_bytes_out",
            "response_time_ms",
            "error_rate_percent",
            "request_count",
            "active_connections",
            "queue_depth"
        ]
    
    def generate_metrics(
        self, 
        start_time: datetime, 
        end_time: datetime, 
        interval_minutes: int = 5
    ) -> List[MetricData]:
        """Generate mock metrics data for the specified time range."""
        metrics = []
        current_time = start_time
        
        while current_time <= end_time:
            for metric_name in self.base_metrics:
                # Generate realistic-looking data with some anomalies
                base_value = self._get_base_value(metric_name)
                variation = random.uniform(-0.2, 0.2)
                
                # Add occasional spikes for error simulation
                if random.random() < 0.05:  # 5% chance of anomaly
                    variation += random.uniform(0.5, 2.0)
                
                value = max(0, base_value * (1 + variation))
                
                metric = MetricData(
                    metric_name=metric_name,
                    value=value,
                    timestamp=current_time,
                    labels={
                        "service": random.choice(["web", "api", "database", "cache"]),
                        "environment": "production",
                        "region": random.choice(["us-east-1", "us-west-2", "eu-west-1"])
                    }
                )
                metrics.append(metric)
            
            current_time += timedelta(minutes=interval_minutes)
        
        return metrics
    
    def _get_base_value(self, metric_name: str) -> float:
        """Get base value for different metric types."""
        base_values = {
            "cpu_usage_percent": 45.0,
            "memory_usage_percent": 60.0,
            "disk_usage_percent": 75.0,
            "network_bytes_in": 1024000.0,
            "network_bytes_out": 512000.0,
            "response_time_ms": 150.0,
            "error_rate_percent": 0.5,
            "request_count": 1000.0,
            "active_connections": 50.0,
            "queue_depth": 10.0
        }
        return base_values.get(metric_name, 50.0)
    
    def analyze_metric_anomalies(
        self, 
        metrics: List[MetricData], 
        threshold_factor: float = 2.0
    ) -> Dict[str, Any]:
        """Analyze metrics for anomalies."""
        if not metrics:
            return {}
        
        # Group metrics by name
        metric_groups = {}
        for metric in metrics:
            if metric.metric_name not in metric_groups:
                metric_groups[metric.metric_name] = []
            metric_groups[metric.metric_name].append(metric.value)
        
        anomalies = {}
        
        for metric_name, values in metric_groups.items():
            if len(values) < 2:
                continue
            
            mean_val = sum(values) / len(values)
            std_dev = (sum((x - mean_val) ** 2 for x in values) / len(values)) ** 0.5
            
            threshold = mean_val + (threshold_factor * std_dev)
            anomaly_count = sum(1 for v in values if v > threshold)
            
            anomalies[metric_name] = {
                "mean": mean_val,
                "std_dev": std_dev,
                "threshold": threshold,
                "anomaly_count": anomaly_count,
                "anomaly_percentage": (anomaly_count / len(values)) * 100
            }
        
        return anomalies


class TracesProcessor:
    """Generate and process mock trace data."""
    
    def __init__(self):
        """Initialize the traces processor."""
        self.operations = [
            "http_request",
            "database_query", 
            "cache_lookup",
            "api_call",
            "file_operation",
            "authentication",
            "authorization",
            "data_processing"
        ]
    
    def generate_traces(
        self, 
        start_time: datetime, 
        end_time: datetime, 
        traces_per_minute: int = 10
    ) -> List[TraceData]:
        """Generate mock trace data."""
        traces = []
        current_time = start_time
        trace_id_counter = 1
        
        time_diff = end_time - start_time
        total_minutes = int(time_diff.total_seconds() / 60)
        
        for minute in range(total_minutes):
            minute_time = start_time + timedelta(minutes=minute)
            
            for _ in range(traces_per_minute):
                operation = random.choice(self.operations)
                
                # Generate realistic durations with occasional slow operations
                base_duration = self._get_base_duration(operation)
                if random.random() < 0.1:  # 10% chance of slow operation
                    duration = base_duration * random.uniform(5, 20)
                    status = "error" if random.random() < 0.3 else "slow"
                else:
                    duration = base_duration * random.uniform(0.5, 2.0)
                    status = "success"
                
                trace = TraceData(
                    trace_id=f"trace_{trace_id_counter:06d}",
                    span_id=f"span_{random.randint(1000, 9999)}",
                    operation_name=operation,
                    duration_ms=duration,
                    status=status,
                    timestamp=minute_time + timedelta(seconds=random.randint(0, 59))
                )
                traces.append(trace)
                trace_id_counter += 1
        
        return traces
    
    def _get_base_duration(self, operation: str) -> float:
        """Get base duration for different operation types."""
        base_durations = {
            "http_request": 50.0,
            "database_query": 100.0,
            "cache_lookup": 5.0,
            "api_call": 200.0,
            "file_operation": 80.0,
            "authentication": 30.0,
            "authorization": 20.0,
            "data_processing": 300.0
        }
        return base_durations.get(operation, 100.0)
    
    def analyze_trace_performance(self, traces: List[TraceData]) -> Dict[str, Any]:
        """Analyze trace performance patterns."""
        if not traces:
            return {}
        
        # Group by operation
        operation_stats = {}
        error_count = 0
        slow_count = 0
        
        for trace in traces:
            if trace.operation_name not in operation_stats:
                operation_stats[trace.operation_name] = {
                    "count": 0,
                    "total_duration": 0,
                    "max_duration": 0,
                    "min_duration": float('inf'),
                    "errors": 0,
                    "slow_operations": 0
                }
            
            stats = operation_stats[trace.operation_name]
            stats["count"] += 1
            stats["total_duration"] += trace.duration_ms
            stats["max_duration"] = max(stats["max_duration"], trace.duration_ms)
            stats["min_duration"] = min(stats["min_duration"], trace.duration_ms)
            
            if trace.status == "error":
                stats["errors"] += 1
                error_count += 1
            elif trace.status == "slow":
                stats["slow_operations"] += 1
                slow_count += 1
        
        # Calculate averages
        for operation, stats in operation_stats.items():
            if stats["count"] > 0:
                stats["avg_duration"] = stats["total_duration"] / stats["count"]
                stats["error_rate"] = (stats["errors"] / stats["count"]) * 100
                stats["slow_rate"] = (stats["slow_operations"] / stats["count"]) * 100
        
        return {
            "total_traces": len(traces),
            "total_errors": error_count,
            "total_slow_operations": slow_count,
            "error_rate": (error_count / len(traces)) * 100,
            "slow_rate": (slow_count / len(traces)) * 100,
            "operation_stats": operation_stats
        } 