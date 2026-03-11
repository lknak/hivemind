"""
HiveAgent - An agent that runs experiments in an isolated git worktree
and coordinates with other agents via pub/sub.
"""
import asyncio
import os
import subprocess
import time
import logging
import shutil
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass

from config import HiveConfig
from pubsub import DiscoveryPubSub, PubSubConfig
from discovery import Discovery, AgentStatus

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Result of running an experiment."""
    success: bool
    val_bpb: Optional[float] = None
    training_seconds: float = 0.0
    peak_vram_mb: float = 0.0
    mfu_percent: float = 0.0
    total_tokens_M: float = 0.0
    num_steps: int = 0
    num_params_M: float = 0.0
    depth: int = 0
    error_message: Optional[str] = None
    claude_output: Optional[str] = None
    diff: Optional[str] = None


class HiveAgent:
    """An agent that runs autoresearch experiments in isolation."""

    def __init__(self, agent_id: int, config: HiveConfig, pubsub: DiscoveryPubSub):
        self.agent_id = agent_id
        self.config = config
        self.pubsub = pubsub

        # Worktree and branch
        self.worktree_path = os.path.abspath(config.get_worktree_path(agent_id))
        self.branch_name = config.get_branch_name(agent_id)

        # State
        self.status = AgentStatus(
            agent_id=str(agent_id),
            worktree_path=self.worktree_path,
            state="idle"
        )

        # Received discoveries (for changes.md)
        self.received_discoveries: List[Discovery] = []

        # Baseline val_bpb (set after first experiment)
        self.baseline_bpb: Optional[float] = None

        logger.info(f"Agent {agent_id} initialized with worktree {self.worktree_path}")

    async def setup_worktree(self) -> bool:
        """Set up the git worktree for this agent."""
        logger.info(f"Agent {self.agent_id}: Setting up worktree at {self.worktree_path}")
        self.status.state = "spawning"
        await self._broadcast_status()

        try:
            loop = asyncio.get_event_loop()

            def run_git_worktree_add():
                return subprocess.run(
                    ["git", "worktree", "add", self.worktree_path, "-b", self.branch_name],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd="autoresearch"
                )

            result = await loop.run_in_executor(None, run_git_worktree_add)

            if result.returncode != 0:
                logger.error(f"Failed to create worktree: {result.stderr}")
                return False

            logger.info(f"Agent {self.agent_id}: Worktree created successfully")
            return True

        except subprocess.TimeoutExpired:
            logger.error(f"Timeout creating worktree for agent {self.agent_id}")
            return False
        except Exception as e:
            logger.error(f"Error setting up worktree: {e}")
            return False

    async def setup_worktree_with_files(self) -> bool:
        """Set up the git worktree and copy necessary files."""
        if not await self.setup_worktree():
            return False
        return True

    async def cleanup_worktree(self) -> None:
        """Remove the git worktree."""
        logger.info(f"Agent {self.agent_id}: Cleaning up worktree")
        try:
            loop = asyncio.get_event_loop()

            def run_git_worktree_remove():
                return subprocess.run(
                    ["git", "worktree", "remove", self.worktree_path, "--force"],
                    capture_output=True,
                    timeout=30
                )

            await loop.run_in_executor(None, run_git_worktree_remove)
        except Exception as e:
            logger.error(f"Error removing worktree: {e}")

    async def build_changes_context(self) -> str:
        """Build the changes context from received discoveries (in memory)."""
        if not self.received_discoveries:
            return "No prior discoveries yet."

        lines = ["# Experiment History", "## Discoveries from Other Agents", ""]

        for disc in sorted(self.received_discoveries, key=lambda d: d.timestamp):
            status = "✅" if disc.is_improvement else "❌"
            improvement_str = f"{disc.improvement:.6f}" if disc.is_improvement else f"-{abs(disc.improvement):.6f}"

            lines.append(f"### Agent {disc.agent_id} - {disc.description}")
            lines.append(f"- **Result**: {status} val_bpb {disc.baseline_bpb:.6f} → {disc.new_bpb:.6f} (Δ: {improvement_str})")
            lines.append(f"- **Type**: {disc.experiment_type}")
            lines.append(f"- **Modified**: {', '.join(disc.modified_parameters)}")
            lines.append(f"- **Time**: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(disc.timestamp))}")
            lines.append("")
            lines.append("```")
            lines.append(f"val_bpb: {disc.new_bpb:.6f}")
            lines.append(f"training_seconds: {disc.training_seconds:.1f}")
            lines.append(f"mfu_percent: {disc.mfu_percent:.2f}")
            lines.append("```")

            if disc.diff:
                lines.append("")
                lines.append("**Changes**:")
                lines.append("```diff")
                lines.append(disc.diff)
                lines.append("```")
            lines.append("")

        return "\n".join(lines)

    async def run_claude_agent(self) -> tuple[Optional[str], str]:
        """Run the Claude Code agent in the worktree. Returns (diff, output)."""
        from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage

        # Debug mode: return mock data without running claude
        if self.config.debug_enabled:
            logger.info(f"Agent {self.agent_id}: DEBUG MODE - Skipping Claude agent")
            mock_diff = """@@ -450,7 +450,7 @@
 # Model size
-DEPTH = 8               # number of transformer layers
+DEPTH = 12              # number of transformer layers (increased for better capacity)
"""
            mock_output = f"Agent {self.agent_id} analysis: Increasing DEPTH from 8 to 12 to improve model capacity and potentially reduce val_bpb."
            return mock_diff, mock_output

        logger.info(f"Agent {self.agent_id}: Running Claude Code agent")

        # Build prompt from received discoveries (in memory)
        changes_context = await self.build_changes_context()
        prompt_content = f"""Analyze the following experiment history and propose improvements.

{changes_context}

Based on this context, modify train.py to improve val_bpb.
"""

        # Run Claude Code with streaming output
        try:
            full_output_lines = []

            # Use claude_agent_sdk for streaming
            async for message in query(
                prompt=prompt_content,
                options=ClaudeAgentOptions(
                    setting_sources=["user"],  # Load from ~/.claude/ only
                    cwd=self.worktree_path,
                ),
            ):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if hasattr(block, "text"):
                            text = block.text
                            full_output_lines.append(text)
                            # Stream to frontend via pubsub
                            try:
                                await self.pubsub.publish_output(
                                    str(self.agent_id), "claude", text
                                )
                            except Exception as e:
                                logger.debug(f"Failed to stream output: {e}")

            full_output = "\n".join(full_output_lines)

            # Get the diff
            loop = asyncio.get_event_loop()

            def run_git_diff():
                return subprocess.run(
                    ["git", "diff", "HEAD", "--", "train.py"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=self.worktree_path
                )

            diff_process = await loop.run_in_executor(None, run_git_diff)
            diff = diff_process.stdout

            return diff if diff.strip() else None, full_output

        except Exception as e:
            logger.error(f"Error running Claude agent: {e}")
            return None, str(e)

    async def run_experiment(self) -> ExperimentResult:
        """Run the training experiment."""

        # Debug mode: return mock result without running experiment
        if self.config.debug_enabled:
            logger.info(f"Agent {self.agent_id}: DEBUG MODE - Skipping experiment execution")
            # Return a mock successful result with slightly varying val_bpb
            import random
            mock_val_bpb = 0.9950 + (self.agent_id * 0.001) + (self.status.experiments_run * 0.0001) + random.uniform(-0.0001, 0.0001)
            return ExperimentResult(
                success=True,
                val_bpb=mock_val_bpb,
                training_seconds=5.0,  # Mock training time
                peak_vram_mb=1000.0,
                mfu_percent=40.0,
                total_tokens_M=500.0,
                num_steps=1000,
                num_params_M=50.0,
                depth=8
            )

        logger.info(f"Agent {self.agent_id}: Running training experiment")

        try:
            # Run train.py using the project's venv uv
            # Use run_in_executor to avoid blocking the event loop
            loop = asyncio.get_event_loop()

            def run_training():
                return subprocess.run(
                    ["../../.venv/bin/python", "train.py"],
                    capture_output=True,
                    text=True,
                    timeout=self.config.time_budget + 120,  # Extra time for startup/shutdown
                    cwd=self.worktree_path
                )

            result = await loop.run_in_executor(None, run_training)

            output = result.stdout + result.stderr

            # Parse the output for val_bpb and other metrics
            metrics = self._parse_training_output(output)

            if metrics and "val_bpb" in metrics:
                return ExperimentResult(
                    success=True,
                    val_bpb=metrics["val_bpb"],
                    training_seconds=metrics.get("training_seconds", 0),
                    peak_vram_mb=metrics.get("peak_vram_mb", 0),
                    mfu_percent=metrics.get("mfu_percent", 0),
                    total_tokens_M=metrics.get("total_tokens_M", 0),
                    num_steps=metrics.get("num_steps", 0),
                    num_params_M=metrics.get("num_params_M", 0),
                    depth=metrics.get("depth", 0),
                    claude_output=output
                )
            else:
                return ExperimentResult(
                    success=False,
                    error_message="Failed to parse training output",
                    claude_output=output
                )

        except subprocess.TimeoutExpired:
            return ExperimentResult(
                success=False,
                error_message=f"Experiment timed out after {self.config.time_budget} seconds"
            )
        except Exception as e:
            return ExperimentResult(
                success=False,
                error_message=str(e)
            )

    def _parse_training_output(self, output: str) -> Optional[dict]:
        """Parse training output to extract metrics."""
        metrics = {}

        for line in output.split("\n"):
            line = line.strip()

            # Parse val_bpb: 0.997900
            if line.startswith("val_bpb:"):
                try:
                    metrics["val_bpb"] = float(line.split(":")[1].strip())
                except (IndexError, ValueError):
                    pass

            # Parse training_seconds: 300.1
            elif line.startswith("training_seconds:"):
                try:
                    metrics["training_seconds"] = float(line.split(":")[1].strip())
                except (IndexError, ValueError):
                    pass

            # Parse peak_vram_mb: 45060.2
            elif line.startswith("peak_vram_mb:"):
                try:
                    metrics["peak_vram_mb"] = float(line.split(":")[1].strip())
                except (IndexError, ValueError):
                    pass

            # Parse mfu_percent: 39.80
            elif line.startswith("mfu_percent:"):
                try:
                    metrics["mfu_percent"] = float(line.split(":")[1].strip())
                except (IndexError, ValueError):
                    pass

            # Parse total_tokens_M: 499.6
            elif line.startswith("total_tokens_M:"):
                try:
                    metrics["total_tokens_M"] = float(line.split(":")[1].strip())
                except (IndexError, ValueError):
                    pass

            # Parse num_steps: 953
            elif line.startswith("num_steps:"):
                try:
                    metrics["num_steps"] = int(line.split(":")[1].strip())
                except (IndexError, ValueError):
                    pass

            # Parse num_params_M: 50.3
            elif line.startswith("num_params_M:"):
                try:
                    metrics["num_params_M"] = float(line.split(":")[1].strip())
                except (IndexError, ValueError):
                    pass

            # Parse depth: 8
            elif line.startswith("depth:"):
                try:
                    metrics["depth"] = int(line.split(":")[1].strip())
                except (IndexError, ValueError):
                    pass

        return metrics if "val_bpb" in metrics else None

    async def create_discovery(self, result: ExperimentResult, diff: Optional[str], claude_output: str) -> Optional[Discovery]:
        """Create a Discovery object from experiment results."""
        if not result.success or result.val_bpb is None:
            return None

        # Calculate improvement
        if self.baseline_bpb is None:
            self.baseline_bpb = result.val_bpb
            return None  # First run establishes baseline, not a discovery

        improvement = self.baseline_bpb - result.val_bpb

        return Discovery(
            agent_id=str(self.agent_id),
            timestamp=time.time(),
            description=claude_output[:500] if claude_output else "Experiment completed",  # Truncate
            diff=diff or "",
            worktree_path=self.worktree_path,
            baseline_bpb=self.baseline_bpb,
            new_bpb=result.val_bpb,
            improvement=improvement,
            training_seconds=result.training_seconds,
            peak_vram_mb=result.peak_vram_mb,
            mfu_percent=result.mfu_percent,
            total_tokens_M=result.total_tokens_M,
            num_steps=result.num_steps,
            num_params_M=result.num_params_M,
            depth=result.depth,
            status="improvement" if improvement > 0 else "degradation"
        )

    async def broadcast_discovery(self, discovery: Discovery) -> None:
        """Broadcast a discovery to other agents."""
        logger.info(f"Agent {self.agent_id}: Broadcasting discovery with improvement {discovery.improvement:.6f}")
        await self.pubsub.publish_discovery(discovery)
        self.status.discoveries_made += 1

    async def _broadcast_status(self) -> None:
        """Broadcast current status."""
        await self.pubsub.publish_status(self.status)

    async def update_status(self, state: str, current_experiment: Optional[str] = None) -> None:
        """Update and broadcast agent status."""
        self.status.state = state
        self.status.current_experiment = current_experiment
        await self._broadcast_status()

    async def receive_discovery(self, discovery: Discovery) -> None:
        """Receive a discovery from another agent."""
        if discovery.agent_id != str(self.agent_id):  # Don't add own discoveries
            self.received_discoveries.append(discovery)
            self.status.discoveries_received += 1
            logger.debug(f"Agent {self.agent_id}: Received discovery from agent {discovery.agent_id}")

    async def run_experiment_loop(self) -> None:
        """Main experiment loop for the agent."""
        logger.info(f"Agent {self.agent_id}: Starting experiment loop")

        # Setup
        if not await self.setup_worktree_with_files():
            logger.error(f"Agent {self.agent_id}: Failed to setup worktree")
            return

        while self.config.experiments_per_agent == -1 or self.status.experiments_run < self.config.experiments_per_agent:
            try:
                # Update status
                await self.update_status("experimenting", f"Experiment #{self.status.experiments_run + 1}")

                # Run Claude agent to make modifications (prompt built from memory)
                diff, claude_output = await self.run_claude_agent()

                if not diff:
                    logger.warning(f"Agent {self.agent_id}: No changes made by Claude")
                    # Still run experiment to establish baseline on first run
                    if self.status.experiments_run == 0:
                        result = await self.run_experiment()
                    else:
                        continue
                else:
                    # Run the experiment
                    result = await self.run_experiment()

                # Process results
                self.status.experiments_run += 1

                if result.success:
                    logger.info(f"Agent {self.agent_id}: Experiment {self.status.experiments_run} completed - val_bpb: {result.val_bpb:.6f}")

                    # Create and broadcast discovery
                    discovery = await self.create_discovery(result, diff, claude_output)
                    if discovery:
                        await self.broadcast_discovery(discovery)

                        # Update baseline if improvement
                        if discovery.is_improvement:
                            self.baseline_bpb = discovery.new_bpb
                            logger.info(f"Agent {self.agent_id}: New baseline established: {self.baseline_bpb:.6f}")

                else:
                    logger.error(f"Agent {self.agent_id}: Experiment failed - {result.error_message}")

                # Reset changes for next iteration (git checkout)
                await self._reset_worktree_changes()

            except Exception as e:
                logger.error(f"Agent {self.agent_id}: Error in experiment loop: {e}")
                self.status.errors.append(str(e))
                await self.update_status("error")
                break

        logger.info(f"Agent {self.agent_id}: Experiment loop completed")

    async def _reset_worktree_changes(self) -> None:
        """Reset the worktree to HEAD (discard changes but keep branch)."""
        try:
            loop = asyncio.get_event_loop()

            def run_git_checkout():
                return subprocess.run(
                    ["git", "checkout", "--", "."],
                    capture_output=True,
                    timeout=10,
                    cwd=self.worktree_path
                )

            await loop.run_in_executor(None, run_git_checkout)
        except Exception as e:
            logger.error(f"Agent {self.agent_id}: Error resetting worktree: {e}")

    async def stop(self) -> None:
        """Stop the agent and clean up."""
        logger.info(f"Agent {self.agent_id}: Stopping")
        await self.update_status("stopped")
        await self.cleanup_worktree()
