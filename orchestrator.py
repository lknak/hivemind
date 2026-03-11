"""
Orchestrator - Spawns and manages multiple HiveAgents.
"""
import asyncio
import logging
import signal
import sys
from typing import List, Optional

from config import HiveConfig
from pubsub import DiscoveryPubSub, PubSubConfig
from hive_agent import HiveAgent
from discovery import Discovery, AgentStatus

logger = logging.getLogger(__name__)


class HiveOrchestrator:
    """Orchestrates multiple HiveAgents and coordinates discovery sharing."""

    def __init__(self, config: HiveConfig):
        self.config = config
        self.pubsub: Optional[DiscoveryPubSub] = None
        self.agents: List[HiveAgent] = []
        self.agent_tasks: List[asyncio.Task] = []
        self.discovery_listener_task: Optional[asyncio.Task] = None
        self.discoveries: List[Discovery] = []
        self.running = False
        self.shutdown_event = asyncio.Event()

        logger.info(f"Orchestrator initialized with {config.num_agents} agents")

    async def setup(self) -> bool:
        """Set up the orchestrator and pub/sub system."""
        logger.info("Setting up orchestrator...")

        # Connect to pub/sub
        pubsub_config = PubSubConfig(
            redis_url=self.config.redis_url,
            max_log_entries=10000
        )
        self.pubsub = DiscoveryPubSub(pubsub_config)
        await self.pubsub.connect()

        # Create agents
        for i in range(self.config.num_agents):
            agent = HiveAgent(i, self.config, self.pubsub)
            self.agents.append(agent)
            logger.info(f"Created agent {i}")

        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        return True

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown_event.set()

    async def start(self) -> None:
        """Start all agents and the discovery listener."""
        logger.info("Starting orchestrator...")
        self.running = True

        # Start discovery listener
        self.discovery_listener_task = asyncio.create_task(self._listen_discoveries())

        # Start all agents
        for agent in self.agents:
            task = asyncio.create_task(agent.run_experiment_loop())
            self.agent_tasks.append(task)
            logger.info(f"Started agent {agent.agent_id}")

        # Wait for shutdown
        await self.shutdown_event.wait()

    async def _listen_discoveries(self) -> None:
        """Listen for discoveries and distribute to agents."""
        logger.info("Starting discovery listener...")

        while self.running:
            try:
                async for discovery in self.pubsub.subscribe_discoveries():
                    logger.info(
                        f"Discovery from agent {discovery.agent_id}: "
                        f"val_bpb {discovery.baseline_bpb:.6f} → {discovery.new_bpb:.6f} "
                        f"(Δ: {discovery.improvement:+.6f})"
                    )

                    # Store discovery
                    self.discoveries.append(discovery)

                    # Distribute to all agents (they'll filter out their own)
                    for agent in self.agents:
                        await agent.receive_discovery(discovery)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in discovery listener: {e}")
                await asyncio.sleep(1)  # Avoid tight loop on error

    async def stop(self) -> None:
        """Stop all agents and clean up."""
        logger.info("Stopping orchestrator...")
        self.running = False

        # If in debug mode with limited experiments, stop agents gracefully
        if self.config.debug_enabled and self.config.experiments_per_agent > 0:
            for agent in self.agents:
                if agent.status.experiments_run >= self.config.experiments_per_agent:
                    await agent.update_status("completed")

        # Cancel discovery listener
        if self.discovery_listener_task:
            self.discovery_listener_task.cancel()
            try:
                await self.discovery_listener_task
            except asyncio.CancelledError:
                pass

        # Stop all agents
        for agent in self.agents:
            await agent.stop()

        # Cancel agent tasks
        for task in self.agent_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Disconnect pub/sub
        if self.pubsub:
            await self.pubsub.disconnect()

        logger.info("Orchestrator stopped")

    def get_status(self) -> dict:
        """Get current status of all agents."""
        return {
            "running": self.running,
            "num_agents": len(self.agents),
            "agents": [agent.status.to_dict() for agent in self.agents],
            "total_discoveries": len(self.discoveries),
            "best_bpb": min((d.new_bpb for d in self.discoveries), default=None),
        }

    def get_discoveries(self) -> List[dict]:
        """Get all discoveries as dictionaries."""
        return [d.to_dict() for d in self.discoveries]


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="HiveMind Orchestrator")
    parser.add_argument("--num-agents", type=int, default=3, help="Number of agents")
    parser.add_argument("--experiments", type=int, default=-1, help="Experiments per agent (-1 = infinite)")
    parser.add_argument("--redis-url", type=str, default="redis://localhost:6379", help="Redis URL")
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level")
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("hivemind.log")
        ]
    )

    # Create config
    config = HiveConfig(
        num_agents=args.num_agents,
        experiments_per_agent=args.experiments,
        redis_url=args.redis_url,
        log_level=args.log_level
    )

    # Create and run orchestrator
    orchestrator = HiveOrchestrator(config)

    try:
        await orchestrator.setup()
        await orchestrator.start()
    finally:
        await orchestrator.stop()


if __name__ == "__main__":
    asyncio.run(main())
