"""
Response models for API endpoints
"""
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime
from models.workflow import WorkflowStatus

class AgentResponse(BaseModel):
    """Standard agent response format"""
    agent_name: str
    status: str  # success, error, processing
    message: str
    data: Dict[str, Any] = {}
    processing_time: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    error_details: Optional[Dict[str, Any]] = None

class WorkflowResponse(BaseModel):
    """Workflow processing response"""
    workflow_id: str
    status: WorkflowStatus
    message: str
    current_step: Optional[str] = None
    progress: float = Field(ge=0.0, le=1.0, default=0.0)
    estimated_completion: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    """Health check response"""
    agent_name: str
    status: str  # healthy, degraded, unhealthy
    uptime: float  # seconds
    version: str
    dependencies: Dict[str, str] = {}  # service_name: status
    metrics: Dict[str, Any] = {}

class VoiceProcessingResponse(BaseModel):
    """Voice processing specific response"""
    transcription: str
    language: str = "en-US"
    confidence: float = Field(ge=0.0, le=1.0)
    intent: Dict[str, Any]
    processing_time: float
    audio_duration: Optional[float] = None

class ActionExecutionResponse(BaseModel):
    """Action execution specific response"""
    action_type: str
    success: bool
    result_data: Dict[str, Any]
    execution_time: float
    external_id: Optional[str] = None
    error_message: Optional[str] = None

class MetricsResponse(BaseModel):
    """System metrics response"""
    total_workflows: int
    recent_workflows_24h: int
    success_rate_24h: float
    average_processing_time: float
    active_workflows: int
    agent_metrics: Dict[str, Any]
    system_health: str

class ErrorResponse(BaseModel):
    """Error response format"""
    error_code: str
    error_message: str
    error_details: Dict[str, Any] = {}
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None

class PaginatedResponse(BaseModel):
    """Paginated response format"""
    items: List[Dict[str, Any]]
    total_count: int
    page: int
    page_size: int
    has_next: bool
    has_previous: bool

