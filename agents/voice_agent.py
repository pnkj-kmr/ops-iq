"""
Enhanced Voice Agent with Ollama integration and improved NLP for Phase 2
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, Optional
import json
import httpx
import asyncio
from datetime import datetime

from agents.base_agent import BaseAgent
from config.settings import settings
from models.workflow import WorkflowStatus, AgentType, Command, Intent
from models.responses import AgentResponse


class VoiceAgent(BaseAgent):
    """Enhanced Voice processing agent"""

    def __init__(self):
        super().__init__("voice_agent", AgentType.VOICE, settings.voice_agent_port)
        self.app = FastAPI(
            title="Voice Agent",
            version="0.2.0",
            description="Voice processing and natural language understanding",
        )

        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Ollama client
        self.ollama_url = settings.ollama_url
        self.ollama_model = settings.ollama_model

        self.setup_routes()

    def setup_routes(self):
        """Setup FastAPI routes"""

        @self.app.on_event("startup")
        async def startup():
            await self.initialize()
            # Test Ollama connection
            await self._test_ollama_connection()

        @self.app.on_event("shutdown")
        async def shutdown():
            await self.cleanup()

        @self.app.get("/health")
        async def health():
            health_resp = await self.health_check()
            # Add Ollama health check
            try:
                await self._test_ollama_connection()
                health_resp.dependencies["ollama"] = "healthy"
            except:
                health_resp.dependencies["ollama"] = "unhealthy"
                health_resp.status = "degraded"
            return health_resp

        @self.app.post("/process_workflow_step")
        async def process_workflow_step(request: Dict[str, Any]):
            """Process workflow step from master agent"""
            workflow_id = request.get("workflow_id")
            data = request.get("data", {})

            if not workflow_id:
                raise HTTPException(status_code=400, detail="workflow_id required")

            return await self.handle_workflow_step(workflow_id, data)

        @self.app.post("/process_audio")
        async def process_audio(audio: UploadFile = File(...)):
            """Direct audio processing endpoint"""
            try:
                audio_data = await audio.read()

                # For Phase 2, we'll mock the audio processing
                # In Phase 3, we'll add actual Whisper integration
                transcription = await self._mock_speech_to_text(audio_data)

                # Process with NLP
                intent_result = await self._extract_intent(transcription)

                return {
                    "transcription": transcription,
                    "intent": intent_result,
                    "status": "success",
                }

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/process_text")
        async def process_text(request: Dict[str, Any]):
            """Direct text processing endpoint"""
            text = request.get("text", "")
            if not text:
                raise HTTPException(status_code=400, detail="text required")

            try:
                intent_result = await self._extract_intent(text)
                return {"text": text, "intent": intent_result, "status": "success"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    async def process_request(
        self, workflow_id: str, request_data: Dict[str, Any]
    ) -> AgentResponse:
        """Process voice/text input and extract intent"""
        try:
            transcription = ""

            # Handle voice input
            if "voice_input" in request_data:
                voice_input = request_data["voice_input"]
                if "transcription" in voice_input:
                    transcription = voice_input["transcription"]
                else:
                    # Mock transcription for Phase 2
                    transcription = await self._mock_speech_to_text(
                        voice_input.get("audio_data", b"")
                    )

            # Handle text input
            elif "text_input" in request_data:
                transcription = request_data["text_input"]

            if not transcription:
                return AgentResponse(
                    agent_name=self.agent_name,
                    status="error",
                    message="No text or voice input provided",
                )

            # Extract intent using Ollama/Mistral
            intent_result = await self._extract_intent(transcription)

            # Create command structure
            command = Command(
                intent=Intent(**intent_result),
                original_text=transcription,
                processed_text=transcription,
                language="en-US",
            )

            return AgentResponse(
                agent_name=self.agent_name,
                status="success",
                message="Voice processing completed",
                data={"transcription": transcription, "command": command.dict()},
            )

        except Exception as e:
            self.logger.error(f"Voice processing failed: {e}")
            return AgentResponse(
                agent_name=self.agent_name,
                status="error",
                message=f"Voice processing failed: {str(e)}",
                error_details={"message": str(e), "type": type(e).__name__},
            )

    async def _test_ollama_connection(self):
        """Test connection to Ollama"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.ollama_url}/api/tags")
                if response.status_code != 200:
                    raise Exception(f"Ollama not responding: {response.status_code}")

                # Test model availability
                tags = response.json()
                model_available = any(
                    self.ollama_model in model.get("name", "")
                    for model in tags.get("models", [])
                )

                if not model_available:
                    raise Exception(f"Model {self.ollama_model} not available")

                self.logger.info("Ollama connection and model verified")

        except Exception as e:
            self.logger.error(f"Ollama connection failed: {e}")
            raise

    async def _extract_intent(self, text: str) -> Dict[str, Any]:
        """Extract intent using Ollama/Mistral"""
        try:
            # System prompt for intent recognition
            system_prompt = """You are an intent recognition system for a voice assistant. 
            Extract the intent and entities from user commands. Return only valid JSON.

            Supported intents:
            - schedule_meeting: Schedule a meeting or appointment
            - send_email: Send an email message
            - set_reminder: Set a reminder or notification
            - search_calendar: Search for calendar events
            - search_email: Search for emails
            - cancel_event: Cancel a calendar event
            - unknown: Cannot determine intent

            Return JSON format:
            {
                "name": "intent_name",
                "confidence": 0.95,
                "entities": {
                    "person": "John Smith",
                    "date": "2025-01-16",
                    "time": "14:00",
                    "subject": "project meeting"
                },
                "parameters": {
                    "duration": 60,
                    "location": "conference room"
                }
            }"""

            user_prompt = f"Extract intent from: '{text}'"

            # Call Ollama
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/chat",
                    json={
                        "model": self.ollama_model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "stream": False,
                        "format": "json",
                    },
                )

                if response.status_code != 200:
                    raise Exception(f"Ollama API error: {response.status_code}")

                result = response.json()
                content = result.get("message", {}).get("content", "{}")

                print(f"result from ollama --- {self.ollama_url}/api/chat -- {result}")

                try:
                    intent_data = json.loads(content)

                    # Validate and set defaults
                    intent_data.setdefault("name", "unknown")
                    intent_data.setdefault("confidence", 0.5)
                    intent_data.setdefault("entities", {})
                    intent_data.setdefault("parameters", {})

                    # Ensure confidence is within bounds
                    intent_data["confidence"] = max(
                        0.0, min(1.0, intent_data["confidence"])
                    )

                    self.logger.info(
                        f"Intent extracted: {intent_data['name']} (confidence: {intent_data['confidence']})"
                    )
                    return intent_data

                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse Ollama response: {content}")
                    return {
                        "name": "unknown",
                        "confidence": 0.0,
                        "entities": {},
                        "parameters": {},
                    }

        except Exception as e:
            self.logger.error(f"Intent extraction failed: {e}")
            return {
                "name": "unknown",
                "confidence": 0.0,
                "entities": {},
                "parameters": {},
            }

    async def _mock_speech_to_text(self, audio_data: bytes) -> str:
        """Mock speech-to-text for Phase 2"""
        # In Phase 3, this will be replaced with actual Whisper integration
        mock_transcriptions = [
            "Schedule a meeting with John tomorrow at 2 PM",
            "Send an email to Sarah about the project update",
            "Set a reminder for the dentist appointment next week",
            "Find my calendar events for today",
            "Cancel the meeting with Bob on Friday",
        ]

        # Simple hash-based selection for consistent results
        import hashlib

        hash_val = int(hashlib.md5(audio_data).hexdigest()[:8], 16)
        selected = mock_transcriptions[hash_val % len(mock_transcriptions)]

        self.logger.info(f"Mock transcription: {selected}")
        return selected


# Create app instance
voice_agent = VoiceAgent()
app = voice_agent.app
