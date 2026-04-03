"""Experiment logging utility"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


class ExperimentLogger:
    """Logger for experiment tracking"""
    
    def __init__(self, experiment_dir: str, experiment_name: Optional[str] = None):
        """
        Initialize experiment logger
        
        Args:
            experiment_dir: Directory to save experiment logs
            experiment_name: Name of the experiment
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        if experiment_name is None:
            experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.experiment_name = experiment_name
        self.log_file = self.experiment_dir / f"{experiment_name}_log.json"
        self.metrics_file = self.experiment_dir / f"{experiment_name}_metrics.json"
        
        self.logs = []
        self.metrics = {}
    
    def log(self, message: str, level: str = "INFO"):
        """
        Log a message
        
        Args:
            message: Message to log
            level: Log level (INFO, WARNING, ERROR)
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message
        }
        self.logs.append(log_entry)
        print(f"[{level}] {message}")
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """
        Log a metric
        
        Args:
            name: Metric name
            value: Metric value
            step: Step number (optional)
        """
        if name not in self.metrics:
            self.metrics[name] = []
        
        metric_entry = {
            "value": value,
            "step": step,
            "timestamp": datetime.now().isoformat()
        }
        self.metrics[name].append(metric_entry)
    
    def log_config(self, config: Dict[str, Any]):
        """
        Log configuration
        
        Args:
            config: Configuration dictionary
        """
        config_file = self.experiment_dir / f"{self.experiment_name}_config.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        self.log(f"Configuration saved to {config_file}")
    
    def save(self):
        """Save logs and metrics to files"""
        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(self.logs, f, indent=2, ensure_ascii=False)
        
        with open(self.metrics_file, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)
        
        self.log(f"Logs saved to {self.log_file}")
        self.log(f"Metrics saved to {self.metrics_file}")

