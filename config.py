"""
Configuration management for HiveMind framework.
"""
import os
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class HiveConfig:
    """Configuration for the HiveMind framework."""

    # Agent configuration
    num_agents: int = 3
    experiments_per_agent: int = -1  # -1 = run indefinitely

    # Redis configuration
    redis_url: str = "redis://localhost:6379"

    # Autoresearch parameters
    time_budget: int = 300  # 5 minutes per experiment

    # HiveMind-specific
    discovery_broadcast_delay: float = 0.0  # Immediate = 0.0
    max_pending_discoveries: int = 100  # Buffer size per agent

    # Worktree configuration
    worktree_base_dir: str = "./worktrees"  # Base directory for agent worktrees

    # Logging configuration
    log_level: str = "INFO"
    log_file: str = "hivemind.log"

    # Dashboard configuration
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 8000

    # Debug mode
    debug_enabled: bool = False  # When True, skip claude and experiment execution

    def __post_init__(self):
        # Create worktrees directory if it doesn't exist
        os.makedirs(self.worktree_base_dir, exist_ok=True)

    def get_worktree_path(self, agent_id: int) -> str:
        """Get the worktree path for a specific agent."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.worktree_base_dir, f"agent{agent_id}_{timestamp}")

    def get_branch_name(self, agent_id: int) -> str:
        """Get the git branch name for a specific agent."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Millisecond precision
        return f"hivemind/agent{agent_id}_{timestamp}"


# Default configuration instance
DEFAULT_CONFIG = HiveConfig()
