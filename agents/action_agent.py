"""
Enhanced Action Agent with external system integration framework for Phase 2
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List
import asyncio
from datetime import datetime, timedelta

from agents.base_agent import BaseAgent
from config.settings import settings
from models.workflow import WorkflowStatus, AgentType
from models.responses import AgentResponse


class ActionAgent(BaseAgent):
    """Enhanced Action execution agent"""

    def __init__(self):
        super().__init__("action_agent", AgentType.ACTION, settings.action_agent_port)
        self.app = FastAPI(
            title="Action Agent",
            version="0.2.0",
            description="External system integration and action execution",
        )

        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Supported actions
        self.action_handlers = {
            "schedule_meeting": self._handle_schedule_meeting,
            "send_email": self._handle_send_email,
            "set_reminder": self._handle_set_reminder,
            "search_calendar": self._handle_search_calendar,
            "search_email": self._handle_search_email,
            "cancel_event": self._handle_cancel_event,
        }

        self.setup_routes()

    def setup_routes(self):
        """Setup FastAPI routes"""

        @self.app.on_event("startup")
        async def startup():
            await self.initialize()
            # Test external service connections
            await self._test_external_connections()

        @self.app.on_event("shutdown")
        async def shutdown():
            await self.cleanup()

        @self.app.get("/health")
        async def health():
            health_resp = await self.health_check()
            # Add external service health checks
            health_resp.dependencies.update(await self._check_external_services())
            return health_resp

        @self.app.post("/process_workflow_step")
        async def process_workflow_step(request: Dict[str, Any]):
            """Process workflow step from master agent"""
            workflow_id = request.get("workflow_id")
            data = request.get("data", {})
            print(f"pknj ---> action agent received the data --> {data}")

            if not workflow_id:
                raise HTTPException(status_code=400, detail="workflow_id required")

            return await self.handle_workflow_step(workflow_id, data)

        @self.app.post("/execute_action")
        async def execute_action(request: Dict[str, Any]):
            """Direct action execution endpoint"""
            action_type = request.get("action_type")
            parameters = request.get("parameters", {})

            if not action_type:
                raise HTTPException(status_code=400, detail="action_type required")

            try:
                result = await self._execute_action(action_type, parameters)
                return {
                    "action_type": action_type,
                    "result": result,
                    "status": "success",
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/supported_actions")
        async def get_supported_actions():
            """Get list of supported actions"""
            return {
                "actions": list(self.action_handlers.keys()),
                "descriptions": {
                    "schedule_meeting": "Schedule a meeting or appointment",
                    "send_email": "Send an email message",
                    "set_reminder": "Set a reminder or notification",
                    "search_calendar": "Search for calendar events",
                    "search_email": "Search for emails",
                    "cancel_event": "Cancel a calendar event",
                },
            }

    async def process_request(
        self, workflow_id: str, request_data: Dict[str, Any]
    ) -> AgentResponse:
        """Process action execution request"""
        try:
            command = request_data.get("command", {})
            if not command:
                return AgentResponse(
                    agent_name=self.agent_name,
                    status="error",
                    message="No command provided",
                )

            intent = command.get("intent", {})
            intent_name = intent.get("name", "unknown")

            if intent_name not in self.action_handlers:
                return AgentResponse(
                    agent_name=self.agent_name,
                    status="error",
                    message=f"Unsupported action: {intent_name}",
                )

            # Execute the action
            result = await self._execute_action(intent_name, intent)

            return AgentResponse(
                agent_name=self.agent_name,
                status="success",
                message="Action executed successfully",
                data={"action_type": intent_name, "result": result},
            )

        except Exception as e:
            self.logger.error(f"Action execution failed: {e}")
            return AgentResponse(
                agent_name=self.agent_name,
                status="error",
                message=f"Action execution failed: {str(e)}",
                error_details={"message": str(e), "type": type(e).__name__},
            )

    async def _execute_action(
        self, action_type: str, intent_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute specific action based on type"""
        handler = self.action_handlers.get(action_type)
        if not handler:
            raise ValueError(f"No handler for action: {action_type}")

        return await handler(intent_data)

    async def _handle_schedule_meeting(
        self, intent_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle meeting scheduling"""
        # Extract entities
        entities = intent_data.get("entities", {})
        parameters = intent_data.get("parameters", {})

        # For Phase 2, we'll mock the meeting creation
        # In Phase 4, we'll add actual Google Calendar/Outlook integration
        meeting_data = {
            "title": entities.get("subject", "Meeting"),
            "attendees": [entities.get("person", "Unknown")],
            "start_time": entities.get("date", "2025-01-16")
            + "T"
            + entities.get("time", "14:00:00"),
            "duration": parameters.get("duration", 60),
            "location": parameters.get("location", "TBD"),
        }

        # Mock calendar API call
        await asyncio.sleep(0.5)  # Simulate API call delay

        result = {
            "meeting_id": f"meeting_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "calendar_link": "https://calendar.google.com/mock-link",
            "status": "created",
            "meeting_data": meeting_data,
        }

        self.logger.info(f"Meeting scheduled: {result['meeting_id']}")
        return result

    async def _handle_send_email(self, intent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle email sending"""
        entities = intent_data.get("entities", {})
        parameters = intent_data.get("parameters", {})

        # For Phase 2, we'll mock the email sending
        # In Phase 4, we'll add actual Gmail/Outlook integration
        email_data = {
            "to": [entities.get("person", "unknown@example.com")],
            "subject": entities.get("subject", "Subject from voice command"),
            "body": parameters.get("body", "This email was sent via voice command."),
            "priority": parameters.get("priority", "normal"),
        }

        # Mock email API call
        await asyncio.sleep(0.3)  # Simulate API call delay

        result = {
            "email_id": f"email_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "status": "sent",
            "email_data": email_data,
        }

        self.logger.info(f"Email sent: {result['email_id']}")
        return result

    async def _handle_set_reminder(self, intent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle reminder creation"""
        entities = intent_data.get("entities", {})
        parameters = intent_data.get("parameters", {})

        # For Phase 2, we'll mock the reminder creation
        reminder_data = {
            "title": entities.get("subject", "Reminder"),
            "description": parameters.get(
                "description", "Reminder set via voice command"
            ),
            "reminder_time": entities.get("date", "2025-01-16")
            + "T"
            + entities.get("time", "09:00:00"),
            "repeat": parameters.get("repeat", "none"),
        }

        # Mock reminder API call
        await asyncio.sleep(0.2)  # Simulate API call delay

        result = {
            "reminder_id": f"reminder_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "status": "created",
            "reminder_data": reminder_data,
        }

        self.logger.info(f"Reminder created: {result['reminder_id']}")
        return result

    async def _handle_search_calendar(
        self, intent_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle calendar search"""
        entities = intent_data.get("entities", {})

        # Mock calendar search
        search_results = [
            {
                "event_id": "evt_001",
                "title": "Team Meeting",
                "start_time": "2025-01-16T10:00:00",
                "attendees": ["john@example.com", "sarah@example.com"],
            },
            {
                "event_id": "evt_002",
                "title": "Project Review",
                "start_time": "2025-01-16T15:00:00",
                "attendees": ["bob@example.com"],
            },
        ]

        result = {
            "search_query": entities.get("query", "today"),
            "results": search_results,
            "count": len(search_results),
        }

        self.logger.info(f"Calendar search completed: {result['count']} results")
        return result

    async def _handle_search_email(self, intent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle email search"""
        entities = intent_data.get("entities", {})

        # Mock email search
        search_results = [
            {
                "email_id": "email_001",
                "subject": "Project Update",
                "from": "john@example.com",
                "date": "2025-01-15T14:30:00",
                "snippet": "Here's the latest update on the project...",
            },
            {
                "email_id": "email_002",
                "subject": "Meeting Notes",
                "from": "sarah@example.com",
                "date": "2025-01-15T16:45:00",
                "snippet": "Notes from today's meeting...",
            },
        ]

        result = {
            "search_query": entities.get("query", "recent"),
            "results": search_results,
            "count": len(search_results),
        }

        self.logger.info(f"Email search completed: {result['count']} results")
        return result

    async def _handle_cancel_event(self, intent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle event cancellation"""
        entities = intent_data.get("entities", {})

        # Mock event cancellation
        event_id = f"evt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        result = {
            "event_id": event_id,
            "status": "cancelled",
            "event_title": entities.get("subject", "Unknown Event"),
            "cancellation_time": datetime.now().isoformat(),
        }

        self.logger.info(f"Event cancelled: {event_id}")
        return result

    async def _test_external_connections(self):
        """Test connections to external services"""
        try:
            # For Phase 2, we'll just log that external services would be tested
            # In Phase 4, we'll add actual tests for Google/Microsoft APIs
            self.logger.info("External service connections would be tested here")

        except Exception as e:
            self.logger.error(f"External service test failed: {e}")

    async def _check_external_services(self) -> Dict[str, str]:
        """Check health of external services"""
        # For Phase 2, we'll mock the health checks
        # In Phase 4, we'll add actual health checks for external APIs
        return {
            "google_calendar": "healthy",
            "gmail": "healthy",
            "microsoft_graph": "healthy",
        }


# Create app instance
action_agent = ActionAgent()
app = action_agent.app
