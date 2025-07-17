"""
Enhanced Master Agent with full workflow orchestration for Phase 2
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List
from datetime import datetime, timedelta

from agents.base_agent import BaseAgent
from config.settings import settings
from models.workflow import (
    Workflow,
    WorkflowStatus,
    AgentType,
    VoiceInput,
    Command,
    Intent,
    WorkflowResult,
)
from models.responses import AgentResponse, WorkflowResponse


class MasterAgent(BaseAgent):
    """Enhanced Master agent for workflow orchestration"""

    def __init__(self):
        super().__init__("master_agent", AgentType.MASTER, settings.master_agent_port)
        self.app = FastAPI(
            title="Master Agent",
            version="0.2.0",
            description="Workflow orchestration and management",
        )

        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Agent URLs
        self.voice_agent_url = (
            f"http://{settings.voice_agent_host}:{settings.voice_agent_port}"
        )
        self.action_agent_url = (
            f"http://{settings.action_agent_host}:{settings.action_agent_port}"
        )

        self.setup_routes()

    def setup_routes(self):
        """Setup FastAPI routes"""

        @self.app.on_event("startup")
        async def startup():
            await self.initialize()

        @self.app.on_event("shutdown")
        async def shutdown():
            await self.cleanup()

        @self.app.get("/health")
        async def health():
            return await self.health_check()

        @self.app.post("/workflow/voice", response_model=WorkflowResponse)
        async def create_voice_workflow(
            request: Dict[str, Any], background_tasks: BackgroundTasks
        ):
            """Create new voice workflow"""
            return await self.create_workflow("voice", request, background_tasks)

        @self.app.post("/workflow/text", response_model=WorkflowResponse)
        async def create_text_workflow(
            request: Dict[str, Any], background_tasks: BackgroundTasks
        ):
            """Create new text workflow"""
            return await self.create_workflow("text", request, background_tasks)

        @self.app.get("/workflow/{workflow_id}", response_model=WorkflowResponse)
        async def get_workflow_status(workflow_id: str):
            """Get workflow status"""
            workflow = await Workflow.find_one({"workflow_id": workflow_id})
            if not workflow:
                raise HTTPException(status_code=404, detail="Workflow not found")

            return WorkflowResponse(
                workflow_id=workflow.workflow_id,
                status=workflow.status,
                message=f"Workflow {workflow.status}",
                progress=self._calculate_progress(workflow),
                result=workflow.result.dict() if workflow.result else None,
            )

        @self.app.get("/workflows", response_model=List[WorkflowResponse])
        async def list_workflows(
            user_id: str = None, status: WorkflowStatus = None, limit: int = 20
        ):
            """List workflows with optional filters"""
            query = {}
            if user_id:
                query["user_id"] = user_id
            if status:
                query["status"] = status

            workflows = (
                await Workflow.find(query)
                .sort(-Workflow.created_at)
                .limit(limit)
                .to_list()
            )

            return [
                WorkflowResponse(
                    workflow_id=wf.workflow_id,
                    status=wf.status,
                    message=f"Workflow {wf.status}",
                    progress=self._calculate_progress(wf),
                    result=wf.result.dict() if wf.result else None,
                )
                for wf in workflows
            ]

        @self.app.delete("/workflow/{workflow_id}")
        async def cancel_workflow(workflow_id: str):
            """Cancel a workflow"""
            workflow = await Workflow.find_one({"workflow_id": workflow_id})
            if not workflow:
                raise HTTPException(status_code=404, detail="Workflow not found")

            if workflow.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]:
                raise HTTPException(
                    status_code=400, detail="Cannot cancel completed workflow"
                )

            workflow.status = WorkflowStatus.CANCELLED
            workflow.updated_at = datetime.utcnow()
            await workflow.save()

            return {"message": "Workflow cancelled successfully"}

        @self.app.get("/metrics")
        async def get_metrics():
            """Get system metrics"""
            # Get recent workflows
            total_workflows = await Workflow.count()
            recent_workflows = await Workflow.find(
                {"created_at": {"$gte": datetime.utcnow() - timedelta(hours=24)}}
            ).count()

            successful_workflows = await Workflow.find(
                {
                    "status": WorkflowStatus.COMPLETED,
                    "created_at": {"$gte": datetime.utcnow() - timedelta(hours=24)},
                }
            ).count()

            return {
                "total_workflows": total_workflows,
                "recent_workflows_24h": recent_workflows,
                "success_rate_24h": successful_workflows / recent_workflows
                if recent_workflows > 0
                else 0,
                "agent_metrics": self.metrics,
                "active_workflows": len(self.current_tasks),
            }

    async def create_workflow(
        self,
        input_type: str,
        request: Dict[str, Any],
        background_tasks: BackgroundTasks,
    ) -> WorkflowResponse:
        """Create and start a new workflow"""
        try:
            # Create workflow document
            workflow = Workflow(
                user_id=request.get("user_id", "anonymous"),
                session_id=request.get("session_id"),
                status=WorkflowStatus.PENDING,
                metadata=request.get("metadata", {}),
            )

            # Set input data based on type
            if input_type == "voice":
                if "voice_input" in request:
                    workflow.voice_input = VoiceInput(**request["voice_input"])
            elif input_type == "text":
                workflow.text_input = request.get("text", "")

            # Save workflow
            await workflow.insert()

            # Start processing in background
            background_tasks.add_task(self.process_workflow, workflow.workflow_id)

            return WorkflowResponse(
                workflow_id=workflow.workflow_id,
                status=workflow.status,
                message="Workflow created and processing started",
                progress=0.0,
            )

        except Exception as e:
            self.logger.error(f"Failed to create workflow: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def process_workflow(self, workflow_id: str):
        """Main workflow processing logic"""
        try:
            # Load workflow
            workflow = await Workflow.find_one({"workflow_id": workflow_id})
            if not workflow:
                self.logger.error(f"Workflow {workflow_id} not found")
                return

            # Update status
            workflow.status = WorkflowStatus.PROCESSING
            workflow.updated_at = datetime.utcnow()
            await workflow.save()

            # Step 1: Voice processing (if voice input)
            if workflow.voice_input:
                voice_response = await self.call_agent(
                    self.voice_agent_url,
                    "process_workflow_step",
                    workflow_id,
                    {"voice_input": workflow.voice_input.dict()},
                )

                if voice_response.status != "success":
                    await self._mark_workflow_failed(
                        workflow,
                        "Voice processing failed",
                        voice_response.error_details,
                    )
                    return

                # Update workflow with voice processing results
                if "command" in voice_response.data:
                    command_data = voice_response.data["command"]
                    workflow.command = Command(**command_data)
                    await workflow.save()

            # Step 2: Text processing (if text input)
            elif workflow.text_input:
                # For text input, we can directly call voice agent for NLP
                print(f"processing text input --> {workflow.text_input}")
                voice_response = await self.call_agent(
                    self.voice_agent_url,
                    "process_workflow_step",
                    workflow_id,
                    {"text_input": workflow.text_input},
                )

                if voice_response.status != "success":
                    await self._mark_workflow_failed(
                        workflow, "Text processing failed", voice_response.error_details
                    )
                    return

                if "command" in voice_response.data:
                    command_data = voice_response.data["command"]
                    workflow.command = Command(**command_data)
                    await workflow.save()

            # Step 3: Action execution
            if workflow.command:
                action_response = await self.call_agent(
                    self.action_agent_url,
                    "process_workflow_step",
                    workflow_id,
                    {"command": workflow.command.dict()},
                )

                if action_response.status != "success":
                    await self._mark_workflow_failed(
                        workflow,
                        "Action execution failed",
                        action_response.error_details,
                    )
                    return

                # Update workflow with final results
                workflow.result = WorkflowResult(
                    success=True,
                    message="Workflow completed successfully",
                    data=action_response.data,
                    total_processing_time=sum(
                        step.processing_time or 0 for step in workflow.agent_steps
                    ),
                )
                workflow.status = WorkflowStatus.COMPLETED
                workflow.completed_at = datetime.utcnow()
                workflow.updated_at = datetime.utcnow()
                await workflow.save()

                self.logger.info(f"Workflow {workflow_id} completed successfully")

            else:
                await self._mark_workflow_failed(
                    workflow, "No valid command generated", None
                )

        except Exception as e:
            self.logger.error(f"Workflow processing failed: {e}")
            workflow = await Workflow.find_one({"workflow_id": workflow_id})
            if workflow:
                await self._mark_workflow_failed(
                    workflow, f"Processing error: {str(e)}", {"exception": str(e)}
                )

    async def _mark_workflow_failed(
        self, workflow: Workflow, message: str, error_details: Dict[str, Any]
    ):
        """Mark workflow as failed with error details"""
        workflow.result = WorkflowResult(
            success=False,
            message=message,
            data=error_details or {},
            total_processing_time=sum(
                step.processing_time or 0 for step in workflow.agent_steps
            ),
        )
        workflow.status = WorkflowStatus.FAILED
        workflow.updated_at = datetime.utcnow()
        await workflow.save()
        self.logger.error(f"Workflow {workflow.workflow_id} failed: {message}")

    def _calculate_progress(self, workflow: Workflow) -> float:
        """Calculate workflow progress percentage"""
        if workflow.status == WorkflowStatus.COMPLETED:
            return 1.0
        elif workflow.status == WorkflowStatus.FAILED:
            return 0.0
        elif workflow.status == WorkflowStatus.PENDING:
            return 0.0

        # Calculate based on completed steps
        total_steps = len(workflow.agent_steps)
        if total_steps == 0:
            return 0.1  # Started but no steps yet

        completed_steps = sum(
            1
            for step in workflow.agent_steps
            if step.status == WorkflowStatus.COMPLETED
        )

        return min(
            completed_steps / max(total_steps, 3), 0.9
        )  # Max 90% until fully completed

    async def process_request(
        self, workflow_id: str, request_data: Dict[str, Any]
    ) -> AgentResponse:
        """Process request from other agents"""
        # Master agent doesn't typically receive requests from other agents
        # This is mainly for health checks and metrics
        return AgentResponse(
            agent_name=self.agent_name,
            status="success",
            message="Master agent operational",
            data={"active_workflows": len(self.current_tasks)},
        )


# Create app instance
master_agent = MasterAgent()
app = master_agent.app
