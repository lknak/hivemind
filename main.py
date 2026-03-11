"""
Main entry point for HiveMind.
"""
import asyncio
import argparse
import logging
import sys
import subprocess
from pathlib import Path

from config import HiveConfig
from orchestrator import HiveOrchestrator
from dashboard import app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logger = logging.getLogger(__name__)


async def run_orchestrator(config: HiveConfig):
    """Run the orchestrator."""
    orchestrator = HiveOrchestrator(config)

    try:
        await orchestrator.setup()
        await orchestrator.start()
    finally:
        await orchestrator.stop()


def run_dashboard(host: str, port: int):
    """Run the dashboard server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")


def main():
    parser = argparse.ArgumentParser(
        description="HiveMind - Multi-Agent Research Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Run orchestrator with 3 agents:
    python main.py --orchestrator --num-agents 3

  Run dashboard only:
    python main.py --dashboard

  Run both orchestrator and dashboard:
    python main.py --both --num-agents 3

  Run orchestrator with limited experiments:
    python main.py --orchestrator --num-agents 2 --experiments 10
"""
    )

    parser.add_argument(
        "--mode", "--orchestrator", "--dashboard", "--both",
        choices=["orchestrator", "dashboard", "both"],
        default="orchestrator",
        help="Run mode: orchestrator only, dashboard only, or both"
    )

    parser.add_argument("--num-agents", type=int, default=3, help="Number of agents")
    parser.add_argument("--experiments", type=int, default=-1, help="Experiments per agent (-1 = infinite)")
    parser.add_argument("--redis-url", type=str, default="redis://localhost:6379", help="Redis URL")
    parser.add_argument("--dashboard-host", type=str, default="0.0.0.0", help="Dashboard host")
    parser.add_argument("--dashboard-port", type=int, default=8000, help="Dashboard port")
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (skip claude and experiment execution)")

    args = parser.parse_args()

    # Parse mode from old-style args for compatibility
    if "--orchestrator" in sys.argv:
        args.mode = "orchestrator"
    elif "--dashboard" in sys.argv:
        args.mode = "dashboard"
    elif "--both" in sys.argv:
        args.mode = "both"

    config = HiveConfig(
        num_agents=args.num_agents,
        experiments_per_agent=args.experiments,
        redis_url=args.redis_url,
        log_level=args.log_level,
        debug_enabled=args.debug
    )

    if args.mode == "orchestrator":
        logger.info(f"Starting HiveMind Orchestrator with {args.num_agents} agents")
        asyncio.run(run_orchestrator(config))

    elif args.mode == "dashboard":
        logger.info(f"Starting HiveMind Dashboard on {args.dashboard_host}:{args.dashboard_port}")
        run_dashboard(args.dashboard_host, args.dashboard_port)

    elif args.mode == "both":
        logger.info(f"Starting HiveMind with {args.num_agents} agents and dashboard")

        # Start dashboard in a separate subprocess
        dashboard_process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "dashboard:app",
             "--host", args.dashboard_host,
             "--port", str(args.dashboard_port),
             "--log-level", "info"]
        )

        try:
            # Give dashboard time to start
            import time
            time.sleep(1)
            logger.info(f"Dashboard started on http://{args.dashboard_host}:{args.dashboard_port}")

            # Run orchestrator
            asyncio.run(run_orchestrator(config))
        finally:
            # Stop dashboard
            logger.info("Stopping dashboard...")
            dashboard_process.terminate()
            try:
                dashboard_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                dashboard_process.kill()
                dashboard_process.wait()


if __name__ == "__main__":
    main()
