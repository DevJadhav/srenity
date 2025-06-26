"""Raw log processor for Apache logs in native format."""

import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from loguru import logger

from ..models.schemas import LogEntry, LogLevel


class RawApacheLogProcessor:
    """Process raw Apache log files in native format."""
    
    def __init__(self, log_file_path: Union[str, Path]):
        """Initialize the raw log processor with log file path."""
        self.log_file_path = Path(log_file_path)
        self.log_entries: List[LogEntry] = []
        self.error_patterns: Dict[str, int] = {}
        self._load_and_parse_logs()
    
    def _load_and_parse_logs(self) -> None:
        """Load and parse raw Apache logs."""
        try:
            with open(self.log_file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            
            logger.info(f"Processing {len(lines)} raw log lines from {self.log_file_path}")
            
            for line_id, line in enumerate(lines, 1):
                parsed_entry = self._parse_apache_log_line(line.strip(), line_id)
                if parsed_entry:
                    self.log_entries.append(parsed_entry)
            
            # Analyze error patterns
            self._analyze_error_patterns()
            
            logger.info(f"Successfully parsed {len(self.log_entries)} log entries")
            logger.info(f"Found {len([e for e in self.log_entries if e.level == LogLevel.ERROR])} error entries")
            
        except Exception as e:
            logger.error(f"Error loading raw log file: {e}")
            raise
    
    def _parse_apache_log_line(self, line: str, line_id: int) -> Optional[LogEntry]:
        """Parse a single Apache log line."""
        if not line.strip():
            return None
        
        # Apache log pattern: [Timestamp] [Level] Message
        pattern = r'\[([^\]]+)\] \[([^\]]+)\] (.+)'
        match = re.match(pattern, line)
        
        if not match:
            # Try alternative pattern without level
            pattern2 = r'\[([^\]]+)\] (.+)'
            match2 = re.match(pattern2, line)
            if match2:
                timestamp_str, content = match2.groups()
                level = LogLevel.INFO  # Default level
            else:
                logger.warning(f"Could not parse line {line_id}: {line[:100]}...")
                return None
        else:
            timestamp_str, level_str, content = match.groups()
            # Map Apache log levels to our enum
            level_mapping = {
                'error': LogLevel.ERROR,
                'warn': LogLevel.WARN,
                'warning': LogLevel.WARN,
                'notice': LogLevel.INFO,
                'info': LogLevel.INFO,
                'debug': LogLevel.DEBUG
            }
            level = level_mapping.get(level_str.lower(), LogLevel.INFO)
        
        # Parse timestamp
        try:
            # Handle Apache timestamp format: "Sun Dec 04 04:47:44 2005"
            timestamp = datetime.strptime(timestamp_str, '%a %b %d %H:%M:%S %Y')
        except ValueError:
            try:
                # Alternative timestamp format
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                logger.warning(f"Could not parse timestamp in line {line_id}: {timestamp_str}")
                timestamp = datetime.now()
        
        # Generate event template and event ID based on content
        event_template, event_id = self._extract_event_pattern(content)
        
        return LogEntry(
            line_id=line_id,
            timestamp=timestamp,
            level=level,
            content=content,
            event_id=event_id,
            event_template=event_template
        )
    
    def _extract_event_pattern(self, content: str) -> tuple[str, str]:
        """Extract event pattern and ID from log content."""
        # Common Apache/mod_jk patterns
        patterns = {
            'mod_jk_error': r'mod_jk child workerEnv in error state \d+',
            'worker_init': r'workerEnv\.init\(\) ok',
            'jk2_init': r'jk2_init\(\) Found child \d+ in scoreboard slot \d+',
            'directory_forbidden': r'Directory index forbidden by rule',
            'client_error': r'\[client [^\]]+\]',
            'generic_error': r'error',
            'notice': r'notice'
        }
        
        for pattern_name, pattern in patterns.items():
            if re.search(pattern, content, re.IGNORECASE):
                # Create template by replacing numbers and IPs with placeholders
                template = re.sub(r'\d+', '<NUM>', content)
                template = re.sub(r'\d+\.\d+\.\d+\.\d+', '<IP>', template)
                template = re.sub(r'/\w+/\w+/[\w/]*', '<PATH>', template)
                return template, pattern_name
        
        # Default pattern
        template = re.sub(r'\d+', '<NUM>', content)
        template = re.sub(r'\d+\.\d+\.\d+\.\d+', '<IP>', template)
        return template, 'generic'
    
    def _analyze_error_patterns(self) -> None:
        """Analyze error patterns in the logs."""
        self.error_patterns = {}
        
        for entry in self.log_entries:
            if entry.level == LogLevel.ERROR:
                pattern = entry.event_id
                self.error_patterns[pattern] = self.error_patterns.get(pattern, 0) + 1
    
    def get_logs_by_timeframe(
        self, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[LogEntry]:
        """Get logs within a specific timeframe."""
        return [
            entry for entry in self.log_entries
            if start_time <= entry.timestamp <= end_time
        ]
    
    def get_error_logs(self, timeframe_hours: int = 24) -> List[LogEntry]:
        """Get all error-level logs within the specified timeframe."""
        if not self.log_entries:
            return []
        
        # Calculate timeframe
        end_time = max(entry.timestamp for entry in self.log_entries)
        start_time = end_time - timedelta(hours=timeframe_hours)
        
        return [
            entry for entry in self.log_entries
            if entry.level == LogLevel.ERROR and start_time <= entry.timestamp <= end_time
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive log statistics."""
        if not self.log_entries:
            return {
                "total_entries": 0,
                "total_errors": 0,
                "timespan_hours": 0,
                "error_patterns": {},
                "level_distribution": {}
            }
        
        # Level distribution
        level_counts = {}
        for entry in self.log_entries:
            level_name = entry.level.value
            level_counts[level_name] = level_counts.get(level_name, 0) + 1
        
        # Time range
        timestamps = [entry.timestamp for entry in self.log_entries]
        min_time = min(timestamps)
        max_time = max(timestamps)
        timespan_hours = (max_time - min_time).total_seconds() / 3600
        
        # Error count
        error_count = level_counts.get('error', 0)
        
        return {
            "total_entries": len(self.log_entries),
            "total_errors": error_count,
            "timespan_hours": timespan_hours,
            "error_patterns": self.error_patterns,
            "level_distribution": level_counts,
            "time_range": {
                "start": min_time.isoformat(),
                "end": max_time.isoformat(),
                "duration_hours": timespan_hours
            }
        }
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """Alias for get_statistics for compatibility with CSV processor."""
        return self.get_statistics()
    
    def analyze_error_patterns(self, timeframe_hours: int = 24) -> Dict[str, Any]:
        """Analyze error patterns in the logs."""
        error_logs = self.get_error_logs(timeframe_hours)
        
        if not error_logs:
            return {"total_errors": 0, "patterns": {}, "frequency": {}}
        
        # Count error patterns
        pattern_counts = {}
        hourly_distribution = {}
        
        for log in error_logs:
            # Count by event pattern
            pattern = log.event_id
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
            # Count by hour
            hour_key = log.timestamp.strftime('%Y-%m-%d %H:00')
            hourly_distribution[hour_key] = hourly_distribution.get(hour_key, 0) + 1
        
        return {
            "total_errors": len(error_logs),
            "patterns": pattern_counts,
            "hourly_distribution": hourly_distribution,
            "time_window": f"{timeframe_hours} hours",
            "error_rate": len(error_logs) / max(len(self.log_entries), 1) * 100
        }
    
    def get_incident_context(self, timeframe_hours: int = 1) -> Dict[str, Any]:
        """Get incident context from logs for the specified timeframe."""
        recent_logs = []
        
        if self.log_entries:
            end_time = max(entry.timestamp for entry in self.log_entries)
            start_time = end_time - timedelta(hours=timeframe_hours)
            recent_logs = self.get_logs_by_timeframe(start_time, end_time)
        
        # Analyze recent activity
        error_logs = [log for log in recent_logs if log.level == LogLevel.ERROR]
        warning_logs = [log for log in recent_logs if log.level == LogLevel.WARN]
        
        # Get most common error patterns
        error_patterns = {}
        for log in error_logs:
            pattern = log.event_id
            error_patterns[pattern] = error_patterns.get(pattern, 0) + 1
        
        # Sort patterns by frequency
        top_patterns = sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "timeframe_hours": timeframe_hours,
            "total_logs": len(recent_logs),
            "error_count": len(error_logs),
            "warning_count": len(warning_logs),
            "top_error_patterns": top_patterns,
            "error_rate": len(error_logs) / max(len(recent_logs), 1) * 100,
            "recent_errors": [
                {
                    "timestamp": log.timestamp.isoformat(),
                    "content": log.content,
                    "pattern": log.event_id
                }
                for log in error_logs[-10:]  # Last 10 errors
            ]
        } 