"""
Workflow data models for MongoDB storage
Enhanced for Phase 2 with comprehensive workflow tracking
"""

from beanie import Document
from pydantic import BaseModel, Field, field_serializer
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
import uuid


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentType(str, Enum):
    MASTER = "master"
    VOICE = "voice"
    ACTION = "action"


class VoiceInput(BaseModel):
    """Voice input data structure"""

    audio_file_id: Optional[str] = None  # GridFS ID for audio file
    transcription: Optional[str] = None
    language: str = "en-US"
    duration: Optional[float] = None  # seconds
    sample_rate: Optional[int] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @field_serializer("created_at")
    def serialize_created_at(self, dt: datetime, _info):
        return dt.strftime("%Y-%m-%d %H:%M:%S")  # or dt.isoformat()


class Intent(BaseModel):
    """Parsed intent from voice command"""

    name: str  # e.g., "schedule_meeting", "send_email"
    confidence: float = Field(ge=0.0, le=1.0)
    entities: Dict[str, Any] = {}
    parameters: Dict[str, Any] = {}


class Command(BaseModel):
    """Structured command after intent recognition"""

    intent: Intent
    original_text: str
    processed_text: Optional[str] = None
    language: str = "en-US"
    requires_confirmation: bool = False
    priority: int = 1  # 1-5, where 5 is highest
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @field_serializer("created_at")
    def serialize_created_at(self, dt: datetime, _info):
        return dt.strftime("%Y-%m-%d %H:%M:%S")  # or dt.isoformat()


class AgentStep(BaseModel):
    """Individual agent processing step"""

    step_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_type: AgentType
    agent_name: str
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    status: WorkflowStatus = WorkflowStatus.PENDING

    # Input/Output data
    input_data: Dict[str, Any] = {}
    output_data: Dict[str, Any] = {}

    # Processing metrics
    processing_time: Optional[float] = None  # seconds
    memory_used: Optional[int] = None  # MB
    cpu_used: Optional[float] = None  # percentage

    # Error information
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    retry_count: int = 0

    @field_serializer("start_time")
    def serialize_start_time(self, dt: datetime, _info):
        return dt.strftime("%Y-%m-%d %H:%M:%S")  # or dt.isoformat()


class ActionResult(BaseModel):
    """Result of action execution"""

    action_type: str
    success: bool
    result_data: Dict[str, Any] = {}
    external_ids: Dict[str, str] = {}  # IDs from external systems
    execution_time: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    error_message: Optional[str] = None

    @field_serializer("timestamp")
    def serialize_timestamp(self, dt: datetime, _info):
        return dt.strftime("%Y-%m-%d %H:%M:%S")  # or dt.isoformat()


class WorkflowResult(BaseModel):
    """Final workflow result"""

    success: bool
    message: str
    data: Dict[str, Any] = {}
    actions_taken: List[ActionResult] = []
    total_processing_time: float
    user_feedback: Optional[str] = None


class Workflow(Document):
    """Main workflow document for MongoDB"""

    workflow_id: str = Field(unique=True, default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    session_id: Optional[str] = None

    # Status and timestamps
    status: WorkflowStatus = WorkflowStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    # Input data
    voice_input: Optional[VoiceInput] = None
    text_input: Optional[str] = None  # For text-based commands

    # Processed data
    command: Optional[Command] = None

    # Processing steps
    agent_steps: List[AgentStep] = []

    # Final result
    result: Optional[WorkflowResult] = None

    # Metadata
    metadata: Dict[str, Any] = {}
    tags: List[str] = []

    @field_serializer("created_at")
    def serialize_created_at(self, dt: datetime, _info):
        return dt.strftime("%Y-%m-%d %H:%M:%S")  # or dt.isoformat()

    @field_serializer("updated_at")
    def serialize_updated_at(self, dt: datetime, _info):
        return dt.strftime("%Y-%m-%d %H:%M:%S")  # or dt.isoformat()

    class Settings:
        collection = "workflows"
        indexes = [
            "workflow_id",
            "user_id",
            "status",
            "created_at",
            [("user_id", 1), ("created_at", -1)],
        ]

    def add_agent_step(
        self, agent_type: AgentType, agent_name: str, input_data: Dict[str, Any] = None
    ) -> AgentStep:
        """Add a new agent step to the workflow"""
        step = AgentStep(
            agent_type=agent_type, agent_name=agent_name, input_data=input_data or {}
        )
        self.agent_steps.append(step)
        self.updated_at = datetime.utcnow()
        return step

    def update_agent_step(self, step_id: str, **kwargs):
        """Update an existing agent step"""
        for step in self.agent_steps:
            if step.step_id == step_id:
                for key, value in kwargs.items():
                    if hasattr(step, key):
                        setattr(step, key, value)
                if "status" in kwargs and kwargs["status"] in [
                    WorkflowStatus.COMPLETED,
                    WorkflowStatus.FAILED,
                ]:
                    step.end_time = datetime.utcnow()
                    if step.start_time:
                        step.processing_time = (
                            step.end_time - step.start_time
                        ).total_seconds()
                break
        self.updated_at = datetime.utcnow()

    def get_current_step(self) -> Optional[AgentStep]:
        """Get the currently processing step"""
        for step in self.agent_steps:
            if step.status == WorkflowStatus.PROCESSING:
                return step
        return None

    def get_completed_steps(self) -> List[AgentStep]:
        """Get all completed steps"""
        return [
            step for step in self.agent_steps if step.status == WorkflowStatus.COMPLETED
        ]

    def calculate_total_processing_time(self) -> float:
        """Calculate total processing time across all steps"""
        return sum(step.processing_time or 0 for step in self.agent_steps)
