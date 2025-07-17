"""
Enhanced base agent with comprehensive functionality for Phase 2
"""

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import httpx
import json
from datetime import datetime
import redis.asyncio as redis
from motor.motor_asyncio import AsyncIOMotorClient
from utils.validation_utils import encode_datetimes
from config.settings import settings
from config.logging import get_logger
from models.workflow import Workflow, AgentStep, WorkflowStatus, AgentType
from models.responses import AgentResponse, HealthResponse


class BaseAgent(ABC):
    """Enhanced base class for all agents"""

    def __init__(self, agent_name: str, agent_type: AgentType, port: int):
        self.agent_name = agent_name
        self.agent_type = agent_type
        self.port = port
        self.start_time = time.time()
        self.version = "0.2.0"

        # Logging
        self.logger = get_logger(agent_name)

        # HTTP client for inter-agent communication
        self.http_client = httpx.AsyncClient(timeout=30.0)

        # Redis connection for message queue
        self.redis: Optional[redis.Redis] = None

        # MongoDB connection
        self.db_client: Optional[AsyncIOMotorClient] = None

        # Agent state
        self.is_healthy = True
        self.current_tasks: Dict[str, Dict] = {}
        self.metrics = {
            "requests_processed": 0,
            "requests_failed": 0,
            "average_response_time": 0.0,
            "last_request_time": None,
        }

    async def initialize(self):
        """Initialize agent connections and resources"""
        try:
            # Initialize Redis connection
            self.redis = redis.Redis.from_url(settings.redis_url)
            await self.redis.ping()
            self.logger.info("Redis connection established")

            # Initialize MongoDB connection
            self.db_client = AsyncIOMotorClient(settings.mongodb_url)
            await self.db_client.admin.command("ping")
            self.logger.info("MongoDB connection established")

            # Initialize Beanie for this agent
            await self._init_beanie()

            # Start background tasks
            asyncio.create_task(self._health_monitor())
            asyncio.create_task(self._metrics_collector())

            self.logger.info(f"{self.agent_name} initialized successfully")

        except Exception as e:
            self.logger.error(f"Agent initialization failed: {e}")
            self.is_healthy = False
            raise

    async def _init_beanie(self):
        """Initialize Beanie with workflow models"""
        from beanie import init_beanie

        await init_beanie(database=self.db_client.ops_iq, document_models=[Workflow])

    @abstractmethod
    async def process_request(
        self, workflow_id: str, request_data: Dict[str, Any]
    ) -> AgentResponse:
        """Process incoming request - implement in subclasses"""

        print("=============== not implemented yet to take action ===================")
        pass

    async def handle_workflow_step(
        self, workflow_id: str, input_data: Dict[str, Any]
    ) -> AgentResponse:
        """Handle a workflow step with full tracking"""
        start_time = time.time()

        try:
            print(f"finding workflow .... {workflow_id}")
            # Load workflow
            workflow = await Workflow.find_one({"workflow_id": workflow_id})
            if not workflow:
                raise ValueError(f"Workflow {workflow_id} not found")

            print(f"finding workflow .... found: {workflow_id}")
            # Add agent step
            step = workflow.add_agent_step(
                agent_type=self.agent_type,
                agent_name=self.agent_name,
                input_data=input_data,
            )
            step.status = WorkflowStatus.PROCESSING
            await workflow.save()
            print(f"step next ... {step}")

            # Add to current tasks
            self.current_tasks[workflow_id] = {
                "step_id": step.step_id,
                "start_time": start_time,
                "status": "processing",
            }

            print(f"self.current_tasks ... {self.current_tasks[workflow_id]}")

            # Process the actual request
            print(f"actual request processing...  {input_data}")
            response = await self.process_request(workflow_id, input_data)
            print(f"actual request processing... response {response}")

            # Update step with results
            processing_time = time.time() - start_time
            workflow.update_agent_step(
                step.step_id,
                status=WorkflowStatus.COMPLETED
                if response.status == "success"
                else WorkflowStatus.FAILED,
                output_data=response.data,
                processing_time=processing_time,
                error_message=response.error_details.get("message")
                if response.error_details
                else None,
            )
            print(
                f"status update for workflow... {WorkflowStatus.COMPLETED if response.status == 'success' else WorkflowStatus.FAILED}"
            )
            await workflow.save()

            # Update metrics
            self.metrics["requests_processed"] += 1
            self._update_average_response_time(processing_time)
            self.metrics["last_request_time"] = datetime.utcnow()

            # Remove from current tasks
            self.current_tasks.pop(workflow_id, None)
            print(
                f"popup the task from workflow  total [{len(self.current_tasks)}]--- {workflow_id}"
            )

            return response

        except Exception as e:
            print(f"error if any ---> {e}")
            # Handle errors
            processing_time = time.time() - start_time
            self.metrics["requests_failed"] += 1

            if workflow_id in self.current_tasks:
                step_id = self.current_tasks[workflow_id]["step_id"]
                workflow = await Workflow.find_one({"workflow_id": workflow_id})
                if workflow:
                    workflow.update_agent_step(
                        step_id,
                        status=WorkflowStatus.FAILED,
                        error_message=str(e),
                        processing_time=processing_time,
                    )
                    await workflow.save()

                self.current_tasks.pop(workflow_id, None)

            self.logger.error(f"Workflow step failed: {e}")
            return AgentResponse(
                agent_name=self.agent_name,
                status="error",
                message=f"Request processing failed: {str(e)}",
                processing_time=processing_time,
                error_details={"message": str(e), "type": type(e).__name__},
            )

    async def call_agent(
        self, agent_url: str, endpoint: str, workflow_id: str, data: Dict[str, Any]
    ) -> AgentResponse:
        """Make HTTP call to another agent with workflow tracking"""
        try:
            if "created_at" in data:
                data["created_at"] = str(data["created_at"])

            payload = {
                "workflow_id": workflow_id,
                "data": data,
                "source_agent": self.agent_name,
                "timestamp": str(datetime.utcnow().isoformat()),
            }

            # {'workflow_id': '7924230d-bb2d-4775-90c4-0010d6c0c87e', 'data': {'command': {'intent': {'name': 'search_location', 'confidence': 0.97, 'entities': {'capital': 'Rajasthan', 'name': 'Rajasthan'}, 'parameters': {'location': 'India'}}, 'original_text': "q1) what Rajasthan's capital?", 'processed_text': "q1) what Rajasthan's capital?", 'language': 'en-US', 'requires_confirmation': False, 'priority': 1, 'created_at': datetime.datetime(2025, 7, 17, 12, 49, 5, 792000)}}, 'source_agent': 'master_agent', 'timestamp': '2025-07-17T12:49:05.801231'}

            print(f"####### calling agent -----> {agent_url}/{endpoint} -- {payload}")
            response = await self.http_client.post(
                f"{agent_url}/{endpoint}", json=payload
            )
            response.raise_for_status()

            result = response.json()
            return AgentResponse(**result)

        except Exception as e:
            self.logger.error(f"Agent call failed: {e}")
            return AgentResponse(
                agent_name="unknown",
                status="error",
                message=f"Agent call failed: {str(e)}",
                error_details={"message": str(e), "type": type(e).__name__},
            )

    async def send_message(self, queue_name: str, message: Dict[str, Any]):
        """Send message to Redis queue"""
        try:
            await self.redis.lpush(queue_name, json.dumps(message))
            self.logger.debug(f"Message sent to queue {queue_name}")
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")

    async def receive_message(
        self, queue_name: str, timeout: int = 1
    ) -> Optional[Dict[str, Any]]:
        """Receive message from Redis queue"""
        try:
            result = await self.redis.brpop(queue_name, timeout=timeout)
            if result:
                _, message = result
                return json.loads(message)
            return None
        except Exception as e:
            self.logger.error(f"Failed to receive message: {e}")
            return None

    async def health_check(self) -> HealthResponse:
        """Comprehensive health check"""
        dependencies = {}

        # Check Redis
        try:
            await self.redis.ping()
            dependencies["redis"] = "healthy"
        except:
            dependencies["redis"] = "unhealthy"
            self.is_healthy = False

        # Check MongoDB
        try:
            await self.db_client.admin.command("ping")
            dependencies["mongodb"] = "healthy"
        except:
            dependencies["mongodb"] = "unhealthy"
            self.is_healthy = False

        uptime = time.time() - self.start_time

        return HealthResponse(
            agent_name=self.agent_name,
            status="healthy" if self.is_healthy else "unhealthy",
            uptime=uptime,
            version=self.version,
            dependencies=dependencies,
            metrics=self.metrics,
        )

    async def _health_monitor(self):
        """Background task to monitor agent health"""
        while True:
            try:
                # Reset health status
                self.is_healthy = True

                # Check critical dependencies
                await self.redis.ping()
                await self.db_client.admin.command("ping")

                # Check if agent is overloaded
                if len(self.current_tasks) > 10:  # Configurable threshold
                    self.logger.warning(
                        f"High task load: {len(self.current_tasks)} active tasks"
                    )

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                self.logger.error(f"Health monitor error: {e}")
                self.is_healthy = False
                await asyncio.sleep(10)  # Retry more frequently when unhealthy

    async def _metrics_collector(self):
        """Background task to collect and report metrics"""
        while True:
            try:
                # Publish metrics to Redis for monitoring
                metrics_data = {
                    "agent_name": self.agent_name,
                    "timestamp": datetime.utcnow().isoformat(),
                    "metrics": encode_datetimes(self.metrics),
                    "active_tasks": len(self.current_tasks),
                    "uptime": time.time() - self.start_time,
                }

                # {'agent_name': 'voice_agent', 'timestamp': '2025-07-17T13:08:46.831273', 'metrics': {'requests_processed': 1, 'requests_failed': 0, 'average_response_time': 3.479965925216675, 'last_request_time': datetime.datetime(2025, 7, 17, 13, 6, 55, 673270)}, 'active_tasks': 0, 'uptime': 120.3297860622406}

                print(f"agent_metrics ---> {metrics_data}")

                await self.redis.lpush("agent_metrics", json.dumps(metrics_data))

                await asyncio.sleep(60)  # Report every minute

            except Exception as e:
                self.logger.error(f"Metrics collector error: {e}")
                await asyncio.sleep(60)

    def _update_average_response_time(self, new_time: float):
        """Update rolling average response time"""
        current_avg = self.metrics["average_response_time"]
        processed = self.metrics["requests_processed"]

        if processed == 1:
            self.metrics["average_response_time"] = new_time
        else:
            # Simple moving average
            self.metrics["average_response_time"] = (
                current_avg * (processed - 1) + new_time
            ) / processed

    async def cleanup(self):
        """Cleanup resources"""
        if self.http_client:
            await self.http_client.aclose()
        if self.redis:
            await self.redis.close()
        if self.db_client:
            self.db_client.close()

        self.logger.info(f"{self.agent_name} cleanup completed")
