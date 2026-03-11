"""
Dashboard - FastAPI server for monitoring the HiveMind swarm.
"""
import asyncio
import json
import logging
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvloop

from config import HiveConfig
from pubsub import DiscoveryPubSub, PubSubConfig
from discovery import Discovery, AgentStatus

logger = logging.getLogger(__name__)

app = FastAPI(title="HiveMind Dashboard", version="1.0.0")

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
pubsub: Optional[DiscoveryPubSub] = None
agents: Dict[str, AgentStatus] = {}
discoveries: List[Discovery] = []
log_entries: List[Dict[str, Any]] = []
agent_output: Dict[str, List[Dict[str, Any]]] = {}  # agent_id -> list of output entries
OUTPUT_MAX_ENTRIES = 1000  # Max output entries per agent


@app.on_event("startup")
async def startup_event():
    """Initialize pub/sub connection and start listeners."""
    global pubsub

    config = PubSubConfig(
        redis_url="redis://localhost:6379",
        max_log_entries=10000
    )
    pubsub = DiscoveryPubSub(config)
    await pubsub.connect()

    # Start background tasks
    asyncio.create_task(_listen_discoveries())
    asyncio.create_task(_listen_status())
    asyncio.create_task(_fetch_logs_periodically())
    asyncio.create_task(_listen_agent_output())

    logger.info("Dashboard started")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up pub/sub connection."""
    if pubsub:
        await pubsub.disconnect()
    logger.info("Dashboard stopped")


async def _listen_discoveries():
    """Background task to listen for new discoveries."""
    while True:
        try:
            if pubsub:
                async for discovery in pubsub.subscribe_discoveries():
                    discoveries.append(discovery)
                    logger.info(
                        f"Discovery from agent {discovery.agent_id}: "
                        f"val_bpb {discovery.baseline_bpb:.6f} → {discovery.new_bpb:.6f}"
                    )
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in discovery listener: {e}")
            await asyncio.sleep(1)


async def _listen_status():
    """Background task to listen for agent status updates."""
    while True:
        try:
            if pubsub:
                async for status in pubsub.subscribe_status():
                    agents[status.agent_id] = status
                    logger.debug(f"Status update from agent {status.agent_id}: {status.state}")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in status listener: {e}")
            await asyncio.sleep(1)


async def _fetch_logs_periodically():
    """Background task to fetch logs periodically."""
    global log_entries
    while True:
        try:
            if pubsub:
                new_logs = await pubsub.get_logs(100)
                # Keep last 1000 logs
                log_entries.extend(new_logs)
                log_entries = log_entries[-1000:]
            await asyncio.sleep(5)  # Fetch every 5 seconds
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error fetching logs: {e}")
            await asyncio.sleep(1)


async def _listen_agent_output():
    """Background task to listen for agent output."""
    global agent_output
    while True:
        try:
            if pubsub:
                async for output_msg in pubsub.subscribe_output():
                    agent_id = output_msg.get("agent_id", "unknown")
                    if agent_id not in agent_output:
                        agent_output[agent_id] = []
                    agent_output[agent_id].append(output_msg)
                    # Keep only last N entries per agent
                    agent_output[agent_id] = agent_output[agent_id][-OUTPUT_MAX_ENTRIES:]
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in output listener: {e}")
            await asyncio.sleep(1)


@app.get("/")
async def root():
    """Serve the frontend."""
    frontend_index = Path(__file__).parent / "frontend" / "index.html"
    if frontend_index.exists():
        return FileResponse(str(frontend_index))
    return {"message": "HiveMind Dashboard", "status": "running"}


@app.get("/api/status")
async def get_status():
    """Get overall system status."""
    return {
        "running": True,
        "num_agents": len(agents),
        "total_discoveries": len(discoveries),
        "best_bpb": min((d.new_bpb for d in discoveries), default=None),
        "uptime": time.time(),
    }


@app.get("/api/agents")
async def get_agents():
    """Get all agent statuses."""
    return list(agents.values())


@app.get("/api/agents/{agent_id}")
async def get_agent(agent_id: str):
    """Get a specific agent's status."""
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agents[agent_id].to_dict()


@app.get("/api/discoveries")
async def get_discoveries(limit: int = 100, offset: int = 0):
    """Get discoveries with pagination."""
    # Sort by timestamp descending
    sorted_discoveries = sorted(discoveries, key=lambda d: d.timestamp, reverse=True)
    paginated = sorted_discoveries[offset:offset + limit]
    return {
        "total": len(discoveries),
        "limit": limit,
        "offset": offset,
        "discoveries": [d.to_dict() for d in paginated],
    }


@app.get("/api/discoveries/best")
async def get_best_discoveries(limit: int = 10):
    """Get the best discoveries (lowest val_bpb)."""
    sorted_by_bpb = sorted(discoveries, key=lambda d: d.new_bpb)
    return [d.to_dict() for d in sorted_by_bpb[:limit]]


@app.get("/api/discoveries/latest")
async def get_latest_discoveries(limit: int = 10):
    """Get the most recent discoveries."""
    sorted_by_time = sorted(discoveries, key=lambda d: d.timestamp, reverse=True)
    return [d.to_dict() for d in sorted_by_time[:limit]]


@app.get("/api/logs")
async def get_logs(limit: int = 100):
    """Get recent log entries."""
    return {
        "total": len(log_entries),
        "logs": log_entries[-limit:]
    }


@app.get("/api/metrics/bpb-over-time")
async def get_bpb_over_time():
    """Get val_bpb progression over time."""
    # Sort discoveries by timestamp
    sorted_discoveries = sorted(discoveries, key=lambda d: d.timestamp)

    # Track best val_bpb over time
    best_so_far = float('inf')
    timeline = []

    for d in sorted_discoveries:
        if d.new_bpb < best_so_far:
            best_so_far = d.new_bpb
        timeline.append({
            "timestamp": d.timestamp,
            "val_bpb": d.new_bpb,
            "best_so_far": best_so_far,
            "agent_id": d.agent_id,
            "is_improvement": d.is_improvement,
        })

    return timeline


@app.get("/api/metrics/agent-stats")
async def get_agent_stats():
    """Get per-agent statistics."""
    stats = {}
    for agent_id, agent in agents.items():
        agent_discoveries = [d for d in discoveries if d.agent_id == agent_id]
        improvements = [d for d in agent_discoveries if d.is_improvement]

        stats[agent_id] = {
            "experiments_run": agent.experiments_run,
            "discoveries_made": len(agent_discoveries),
            "improvements": len(improvements),
            "discoveries_received": agent.discoveries_received,
            "best_val_bpb": min((d.new_bpb for d in agent_discoveries), default=None),
            "state": agent.state,
        }

    return stats


@app.get("/api/timeline")
async def get_timeline(hours: int = 24):
    """Get a timeline of events."""
    cutoff = time.time() - (hours * 3600)

    events = []

    # Add discovery events
    for d in discoveries:
        if d.timestamp >= cutoff:
            events.append({
                "timestamp": d.timestamp,
                "type": "discovery",
                "agent_id": d.agent_id,
                "data": {
                    "val_bpb": d.new_bpb,
                    "improvement": d.improvement,
                    "description": d.description[:100],
                }
            })

    # Sort by timestamp
    events.sort(key=lambda e: e["timestamp"])

    return events


@app.get("/api/agent-output/{agent_id}")
async def get_agent_output(agent_id: str, limit: int = 1000):
    """Get recent output from a specific agent."""
    if agent_id not in agent_output:
        return {"agent_id": agent_id, "output": []}
    output = agent_output[agent_id][-limit:]
    return {"agent_id": agent_id, "output": output}


@app.get("/api/agent-output/{agent_id}/stream")
async def stream_agent_output(agent_id: str):
    """Stream agent output in real-time via Server-Sent Events."""
    async def event_generator():
        # Send existing output first
        if agent_id in agent_output:
            for entry in agent_output[agent_id][-1000:]:
                yield f"data: {entry.to_json() if hasattr(entry, 'to_json') else json.dumps(entry)}\n\n"

        # Keep connection alive and send new output
        while True:
            try:
                # Check for new output
                if agent_id in agent_output:
                    # Get the last entry we sent (if any) by tracking position
                    # For simplicity, just send everything periodically
                    pass
                await asyncio.sleep(0.5)  # Check every 500ms
                # Heartbeat to keep connection alive
                yield f": keepalive\n\n"
            except asyncio.CancelledError:
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# Serve frontend static files
frontend_dir = Path(__file__).parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HiveMind Dashboard")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
