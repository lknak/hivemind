"""
Discovery data structures for HiveMind framework.
"""
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional
import json


@dataclass
class Discovery:
    """Structured discovery object combining human-readable and machine-readable information."""

    agent_id: str
    timestamp: float

    # Human-readable description
    description: str  # e.g., "Increased DEPTH to 12, improved val_bpb by 0.002"

    # Machine-readable details
    diff: str  # Git diff of train.py changes
    worktree_path: str  # Path to the worktree containing the change

    # Metrics
    baseline_bpb: float
    new_bpb: float
    improvement: float  # baseline - new (positive = improvement)

    # Additional metrics from training output
    training_seconds: float = 0.0
    peak_vram_mb: float = 0.0
    mfu_percent: float = 0.0
    total_tokens_M: float = 0.0
    num_steps: int = 0
    num_params_M: float = 0.0
    depth: int = 0

    # Metadata
    experiment_type: str = "unknown"  # "architecture", "hyperparameter", "optimizer", etc.
    modified_parameters: List[str] = None  # e.g., ["DEPTH", "MATRIX_LR"]

    # Status
    status: str = "improvement"  # "improvement", "degradation", "crash"

    def __post_init__(self):
        if self.modified_parameters is None:
            self.modified_parameters = []

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict) -> "Discovery":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "Discovery":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))

    @property
    def is_improvement(self) -> bool:
        """Check if this discovery represents an improvement."""
        return self.improvement > 0


@dataclass
class AgentStatus:
    """Status information for an agent."""

    agent_id: str
    worktree_path: str
    state: str  # "idle", "spawning", "experimenting", "broadcasting", "error"
    current_experiment: Optional[str] = None
    last_discovery_time: Optional[float] = None
    experiments_run: int = 0
    discoveries_made: int = 0
    discoveries_received: int = 0
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict) -> "AgentStatus":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "AgentStatus":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))
