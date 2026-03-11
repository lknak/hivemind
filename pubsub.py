"""
Pub/Sub message bus using Redis for inter-agent communication.
"""
import asyncio
import json
import time
import logging
from typing import AsyncIterator, Dict, Any, Optional, List
from dataclasses import dataclass

import redis.asyncio as aioredis

from discovery import Discovery, AgentStatus

logger = logging.getLogger(__name__)


@dataclass
class PubSubConfig:
    """Configuration for the pub/sub system."""
    redis_url: str = "redis://localhost:6379"
    discoveries_channel: str = "hivemind:discoveries"
    status_channel: str = "hivemind:status"
    log_channel: str = "hivemind:logs"
    output_channel: str = "hivemind:output"  # Channel for streaming agent output
    max_log_entries: int = 10000  # Max entries to keep in log queue


class DiscoveryPubSub:
    """Redis-based pub/sub for discovery broadcasting."""

    def __init__(self, config: Optional[PubSubConfig] = None):
        self.config = config or PubSubConfig()
        self.redis: Optional[aioredis.Redis] = None
        self._connected = False

    async def connect(self):
        """Establish connection to Redis."""
        if self._connected:
            return
        self.redis = await aioredis.from_url(
            self.config.redis_url,
            encoding="utf-8",
            decode_responses=True
        )
        self._connected = True
        logger.info(f"Connected to Redis at {self.config.redis_url}")

    async def disconnect(self):
        """Close Redis connection."""
        if self.redis and self._connected:
            await self.redis.close()
            self._connected = False
            logger.info("Disconnected from Redis")

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()

    async def publish_discovery(self, discovery: Discovery) -> int:
        """Publish a discovery to all subscribers. Returns number of subscribers."""
        if not self._connected:
            await self.connect()

        message = discovery.to_json()
        count = await self.redis.publish(self.config.discoveries_channel, message)
        logger.debug(f"Published discovery from agent {discovery.agent_id}, {count} subscribers")
        return count

    async def publish_status(self, status: AgentStatus) -> int:
        """Publish agent status update."""
        if not self._connected:
            await self.connect()

        message = status.to_json()
        count = await self.redis.publish(self.config.status_channel, message)
        return count

    async def publish_output(self, agent_id: str, output_type: str, content: str) -> int:
        """Publish agent output (e.g., claude output, experiment output)."""
        if not self._connected:
            await self.connect()

        message = json.dumps({
            "agent_id": agent_id,
            "type": output_type,  # "claude", "experiment", "error"
            "content": content,
            "timestamp": time.time()
        })
        count = await self.redis.publish(self.config.output_channel, message)
        return count

    async def subscribe_output(self) -> AsyncIterator[Dict[str, Any]]:
        """Subscribe to output channel. Yields output dictionaries."""
        if not self._connected:
            await self.connect()

        pubsub = self.redis.pubsub()
        await pubsub.subscribe(self.config.output_channel)

        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    yield json.loads(message["data"])
        finally:
            await pubsub.unsubscribe(self.config.output_channel)
            await pubsub.close()

    async def publish_log(self, log_entry: Dict[str, Any]) -> None:
        """Publish a log entry to the log queue."""
        if not self._connected:
            await self.connect()

        # Add timestamp if not present
        if "timestamp" not in log_entry:
            log_entry["timestamp"] = time.time()

        # Use LPUSH/RPOP pattern for log queue with max length
        await self.redis.lpush(self.config.log_channel, json.dumps(log_entry))
        await self.redis.ltrim(self.config.log_channel, 0, self.config.max_log_entries - 1)

    async def subscribe_discoveries(self) -> AsyncIterator[Discovery]:
        """Subscribe to discoveries channel. Yields Discovery objects."""
        if not self._connected:
            await self.connect()

        pubsub = self.redis.pubsub()
        await pubsub.subscribe(self.config.discoveries_channel)

        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    data = json.loads(message["data"])
                    yield Discovery.from_dict(data)
        finally:
            await pubsub.unsubscribe(self.config.discoveries_channel)
            await pubsub.close()

    async def subscribe_status(self) -> AsyncIterator[AgentStatus]:
        """Subscribe to status channel. Yields AgentStatus objects."""
        if not self._connected:
            await self.connect()

        pubsub = self.redis.pubsub()
        await pubsub.subscribe(self.config.status_channel)

        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    data = json.loads(message["data"])
                    yield AgentStatus.from_dict(data)
        finally:
            await pubsub.unsubscribe(self.config.status_channel)
            await pubsub.close()

    async def get_logs(self, count: int = 100) -> List[Dict[str, Any]]:
        """Get recent log entries."""
        if not self._connected:
            await self.connect()

        entries = await self.redis.lrange(self.config.log_channel, 0, count - 1)
        return [json.loads(e) for e in entries]

    async def get_all_discoveries(self) -> List[Discovery]:
        """Get all stored discoveries from Redis (if stored in a separate key)."""
        # This would require storing discoveries separately, not just publishing
        # For now, return empty - caller should maintain their own cache
        return []

    async def heartbeat(self, agent_id: str) -> None:
        """Send a heartbeat for an agent."""
        if not self._connected:
            await self.connect()

        key = f"hivemind:heartbeat:{agent_id}"
        await self.redis.setex(key, 60, str(time.time()))  # 60 second TTL

    async def is_agent_alive(self, agent_id: str) -> bool:
        """Check if an agent has sent a recent heartbeat."""
        if not self._connected:
            await self.connect()

        key = f"hivemind:heartbeat:{agent_id}"
        return await self.redis.exists(key) == 1
