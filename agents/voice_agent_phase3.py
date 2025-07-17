"""
Enhanced Voice Agent for Phase 3 - Real Voice Processing Pipeline
Integrates Whisper, VAD, and enhanced Mistral processing
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, Optional, List
import json
import httpx
import asyncio
from datetime import datetime
import tempfile
import os

from agents.base_agent import BaseAgent
from config.settings import settings
from models.workflow import WorkflowStatus, AgentType, Command, Intent, VoiceInput
from models.responses import AgentResponse, VoiceProcessingResponse
from services.whisper_service import whisper_service, TranscriptionResult
from services.audio_service import audio_service, AudioMetadata
from services.voice_activity_detector import voice_activity_detector, VoiceSegment

class EnhancedVoiceAgent(BaseAgent):
    """Enhanced Voice processing agent with real audio processing"""
    
    def __init__(self):
        super().__init__("voice_agent_v3", AgentType.VOICE, settings.voice_agent_port)
        self.app = FastAPI(
            title="Enhanced Voice Agent",
            version="0.3.0",
            description="Real voice processing with Whisper and advanced NLP"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Ollama client settings
        self.ollama_url = settings.ollama_url
        self.ollama_model = settings.ollama_model
        
        # Voice processing settings
        self.max_audio_duration = 300  # 5 minutes
        self.supported_languages = ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"]
        self.confidence_threshold = 0.7  # Minimum confidence for processing
        
        self.setup_routes()
    
    def setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.on_event("startup")
        async def startup():
            await self.initialize()
            # Initialize Whisper and audio services
            await self._initialize_voice_services()
        
        @self.app.on_event("shutdown")
        async def shutdown():
            await self.cleanup()
        
        @self.app.get("/health")
        async def health():
            health_resp = await self.health_check()
            # Add voice service health checks
            voice_health = await self._check_voice_services_health()
            health_resp.dependencies.update(voice_health)
            return health_resp
        
        @self.app.post("/process_workflow_step")
        async def process_workflow_step(request: Dict[str, Any]):
            """Process workflow step from master agent"""
            workflow_id = request.get("workflow_id")
            data = request.get("data", {})
            
            if not workflow_id:
                raise HTTPException(status_code=400, detail="workflow_id required")
            
            return await self.handle_workflow_step(workflow_id, data)
        
        @self.app.post("/upload_audio", response_model=VoiceProcessingResponse)
        async def upload_audio(
            audio_file: UploadFile = File(...),
            user_id: Optional[str] = Form(None),
            workflow_id: Optional[str] = Form(None),
            language: Optional[str] = Form(None),
            enable_vad: bool = Form(True),
            include_segments: bool = Form(False)
        ):
            """Upload and process audio file"""
            try:
                # Read audio file
                audio_data = await audio_file.read()
                
                # Process the audio
                result = await self._process_audio_upload(
                    audio_data=audio_data,
                    filename=audio_file.filename or "upload.wav",
                    content_type=audio_file.content_type or "audio/wav",
                    user_id=user_id,
                    workflow_id=workflow_id,
                    language=language,
                    enable_vad=enable_vad,
                    include_segments=include_segments
                )
                
                return result
                
            except Exception as e:
                self.logger.error(f"Audio upload processing failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/process_text_advanced")
        async def process_text_advanced(request: Dict[str, Any]):
            """Advanced text processing with enhanced NLP"""
            text = request.get("text", "")
            if not text:
                raise HTTPException(status_code=400, detail="text required")
            
            try:
                # Enhanced text processing
                result = await self._process_text_enhanced(
                    text=text,
                    context=request.get("context", {}),
                    user_id=request.get("user_id"),
                    language=request.get("language", "en")
                )
                
                return result
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/supported_languages")
        async def get_supported_languages():
            """Get supported languages for voice processing"""
            return {
                "languages": self.supported_languages,
                "whisper_model": whisper_service.model_size,
                "vad_enabled": True
            }
        
        @self.app.get("/audio_file/{file_id}")
        async def get_audio_file(file_id: str):
            """Retrieve audio file metadata"""
            try:
                metadata = await audio_service.get_audio_metadata(file_id)
                if not metadata:
                    raise HTTPException(status_code=404, detail="Audio file not found")
                
                return metadata
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/voice_segments/{file_id}")
        async def get_voice_segments(file_id: str):
            """Get voice activity segments for an audio file"""
            try:
                # This would require storing VAD results with the audio file
                # For now, return a placeholder response
                return {
                    "file_id": file_id,
                    "segments": [],
                    "message": "Voice segments not stored - process audio to generate"
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
    
    async def _initialize_voice_services(self):
        """Initialize Whisper and audio services"""
        try:
            # Initialize Whisper
            await whisper_service.initialize()
            self.logger.info("Whisper service initialized")
            
            # Initialize audio service
            await audio_service.initialize(self.db_client)
            self.logger.info("Audio service initialized")
            
        except Exception as e:
            self.logger.error(f"Voice services initialization failed: {e}")
            raise
    
    async def _check_voice_services_health(self) -> Dict[str, str]:
        """Check health of voice processing services"""
        health_status = {}
        
        try:
            # Check Whisper
            whisper_health = await whisper_service.health_check()
            health_status["whisper"] = whisper_health.get("status", "unknown")
            
            # Check audio service
            audio_health = await audio_service.health_check()
            health_status["audio_service"] = audio_health.get("status", "unknown")
            
            # Check Ollama
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.ollama_url}/api/tags", timeout=5.0)
                health_status["ollama"] = "healthy" if response.status_code == 200 else "unhealthy"
            
        except Exception as e:
            self.logger.error(f"Voice services health check failed: {e}")
            health_status["voice_services"] = "unhealthy"
        
        return health_status
    
    async def process_request(self, workflow_id: str, request_data: Dict[str, Any]) -> AgentResponse:
        """Process voice/text input with enhanced capabilities"""
        try:
            transcription = ""
            audio_metadata = None
            voice_segments = []
            
            # Handle voice input with real processing
            if "voice_input" in request_data:
                voice_input = request_data["voice_input"]
                
                if "audio_file_id" in voice_input:
                    # Process stored audio file
                    result = await self._process_stored_audio(
                        voice_input["audio_file_id"],
                        voice_input.get("language")
                    )
                    transcription = result["transcription"]
                    audio_metadata = result.get("metadata")
                    voice_segments = result.get("segments", [])
                    
                elif "transcription" in voice_input:
                    # Use provided transcription
                    transcription = voice_input["transcription"]
                
                else:
                    # Handle raw audio data (if provided)
                    audio_data = voice_input.get("audio_data", b"")
                    if audio_data:
                        result = await self._process_raw_audio(
                            audio_data,
                            voice_input.get("language")
                        )
                        transcription = result["transcription"]
                        voice_segments = result.get("segments", [])
            
            # Handle text input
            elif "text_input" in request_data:
                transcription = request_data["text_input"]
            
            if not transcription:
                return AgentResponse(
                    agent_name=self.agent_name,
                    status="error",
                    message="No text or voice input provided"
                )
            
            # Enhanced intent recognition with context
            context = {
                "workflow_id": workflow_id,
                "user_id": request_data.get("user_id"),
                "session_context": request_data.get("context", {}),
                "voice_segments": len(voice_segments),
                "audio_duration": sum(seg.get("duration", 0) for seg in voice_segments)
            }
            
            intent_result = await self._extract_intent_enhanced(transcription, context)
            
            # Create enhanced command structure
            command = Command(
                intent=Intent(**intent_result),
                original_text=transcription,
                processed_text=transcription,
                language=intent_result.get("language", "en")
            )
            
            # Prepare response data
            response_data = {
                "transcription": transcription,
                "command": command.dict(),
                "processing_info": {
                    "voice_segments_detected": len(voice_segments),
                    "total_audio_duration": sum(seg.get("duration", 0) for seg in voice_segments),
                    "confidence_score": intent_result.get("confidence", 0.0),
                    "language_detected": intent_result.get("language", "unknown")
                }
            }
            
            if audio_metadata:
                response_data["audio_metadata"] = audio_metadata
            
            return AgentResponse(
                agent_name=self.agent_name,
                status="success",
                message="Voice processing completed with enhanced pipeline",
                data=response_data
            )
            
        except Exception as e:
            self.logger.error(f"Enhanced voice processing failed: {e}")
            return AgentResponse(
                agent_name=self.agent_name,
                status="error",
                message=f"Voice processing failed: {str(e)}",
                error_details={"message": str(e), "type": type(e).__name__}
            )
    
    async def _process_audio_upload(
        self,
        audio_data: bytes,
        filename: str,
        content_type: str,
        user_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        language: Optional[str] = None,
        enable_vad: bool = True,
        include_segments: bool = False
    ) -> VoiceProcessingResponse:
        """Process uploaded audio file with full pipeline"""
        try:
            # Store audio file
            audio_metadata = await audio_service.upload_audio_file(
                file_data=audio_data,
                filename=filename,
                content_type=content_type,
                user_id=user_id,
                workflow_id=workflow_id
            )
            
            # Process with Whisper
            whisper_options = {
                "language": language,
                "include_segments": include_segments,
                "include_word_timestamps": include_segments
            }
            
            transcription_result = await whisper_service.transcribe_audio(
                audio_data, **whisper_options
            )
            
            # Voice Activity Detection if enabled
            voice_segments = []
            if enable_vad and transcription_result.text:
                try:
                    import librosa
                    import io
                    
                    # Load audio for VAD
                    audio_buffer = io.BytesIO(audio_data)
                    audio_signal, sr = librosa.load(audio_buffer, sr=16000)
                    
                    # Detect voice activity
                    vad_segments = voice_activity_detector.detect_voice_activity(
                        audio_signal, sr, method="combined"
                    )
                    
                    voice_segments = [
                        {
                            "start_time": seg.start_time,
                            "end_time": seg.end_time,
                            "duration": seg.duration,
                            "confidence": seg.confidence
                        }
                        for seg in vad_segments
                    ]
                    
                except Exception as e:
                    self.logger.warning(f"VAD processing failed: {e}")
            
            # Enhanced intent recognition
            context = {
                "audio_duration": audio_metadata.duration,
                "audio_quality": "high" if transcription_result.confidence > 0.8 else "medium",
                "voice_segments": len(voice_segments),
                "user_id": user_id
            }
            
            intent_result = await self._extract_intent_enhanced(
                transcription_result.text, context
            )
            
            return VoiceProcessingResponse(
                transcription=transcription_result.text,
                language=transcription_result.language,
                confidence=transcription_result.confidence,
                intent=intent_result,
                processing_time=transcription_result.processing_time,
                audio_duration=audio_metadata.duration
            )
            
        except Exception as e:
            self.logger.error(f"Audio upload processing failed: {e}")
            raise
    
    async def _process_stored_audio(
        self, 
        audio_file_id: str, 
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process previously stored audio file"""
        try:
            # Retrieve audio file
            audio_data = await audio_service.get_audio_file(audio_file_id)
            if not audio_data:
                raise ValueError(f"Audio file not found: {audio_file_id}")
            
            # Get metadata
            metadata = await audio_service.get_audio_metadata(audio_file_id)
            
            # Transcribe with Whisper
            transcription_result = await whisper_service.transcribe_audio(
                audio_data, language=language
            )
            
            return {
                "transcription": transcription_result.text,
                "confidence": transcription_result.confidence,
                "language": transcription_result.language,
                "metadata": metadata.dict() if metadata else None,
                "processing_time": transcription_result.processing_time
            }
            
        except Exception as e:
            self.logger.error(f"Stored audio processing failed: {e}")
            raise
    
    async def _process_raw_audio(
        self, 
        audio_data: bytes, 
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process raw audio data without storing"""
        try:
            # Transcribe with Whisper
            transcription_result = await whisper_service.transcribe_audio(
                audio_data, language=language, include_segments=True
            )
            
            # Voice Activity Detection
            voice_segments = []
            try:
                import librosa
                import io
                
                audio_buffer = io.BytesIO(audio_data)
                audio_signal, sr = librosa.load(audio_buffer, sr=16000)
                
                vad_segments = voice_activity_detector.detect_voice_activity(
                    audio_signal, sr
                )
                
                voice_segments = [
                    {
                        "start_time": seg.start_time,
                        "end_time": seg.end_time,
                        "duration": seg.duration,
                        "confidence": seg.confidence
                    }
                    for seg in vad_segments
                ]
                
            except Exception as e:
                self.logger.warning(f"VAD processing failed: {e}")
            
            return {
                "transcription": transcription_result.text,
                "confidence": transcription_result.confidence,
                "language": transcription_result.language,
                "segments": voice_segments,
                "processing_time": transcription_result.processing_time
            }
            
        except Exception as e:
            self.logger.error(f"Raw audio processing failed: {e}")
            raise
    
    async def _extract_intent_enhanced(
        self, 
        text: str, 
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Enhanced intent extraction with context awareness"""
        try:
            # Enhanced system prompt with context awareness
            system_prompt = """You are an advanced intent recognition system for a voice assistant. 
            Extract intent and entities from user commands with high accuracy.

            Supported intents:
            - schedule_meeting: Schedule meetings, appointments, calls
            - send_email: Send emails, messages, communications
            - set_reminder: Set reminders, alarms, notifications
            - search_calendar: Search calendar events, check availability
            - search_email: Search emails, find messages
            - cancel_event: Cancel meetings, appointments, events
            - create_task: Create tasks, todos, action items
            - update_task: Update existing tasks or events
            - get_weather: Get weather information
            - make_call: Initiate phone calls
            - unknown: Cannot determine clear intent

            Context considerations:
            - Time references: "tomorrow", "next week", "at 2pm"
            - Person references: names, titles, relationships
            - Location references: places, addresses, "here", "there"
            - Urgency indicators: "urgent", "asap", "when possible"
            - Confirmation phrases: "yes", "no", "confirm", "cancel"

            Return JSON format:
            {
                "name": "intent_name",
                "confidence": 0.95,
                "entities": {
                    "person": ["John Smith", "Sarah"],
                    "date": "2025-01-16",
                    "time": "14:00",
                    "subject": "project meeting",
                    "location": "conference room",
                    "urgency": "normal"
                },
                "parameters": {
                    "duration": 60,
                    "priority": "normal",
                    "confirmed": false
                },
                "language": "en",
                "reasoning": "Brief explanation of intent detection"
            }"""
            
            # Add context information to the prompt
            context_info = ""
            if context:
                context_items = []
                if context.get("audio_duration"):
                    context_items.append(f"Audio duration: {context['audio_duration']:.1f}s")
                if context.get("voice_segments"):
                    context_items.append(f"Voice segments: {context['voice_segments']}")
                if context.get("user_id"):
                    context_items.append(f"User context available")
                
                if context_items:
                    context_info = f"\nContext: {', '.join(context_items)}"
            
            user_prompt = f"Extract intent from: '{text}'{context_info}"
            
            # Call Ollama with enhanced prompt
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/chat",
                    json={
                        "model": self.ollama_model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "stream": False,
                        "format": "json",
                        "options": {
                            "temperature": 0.1,  # Lower temperature for more consistent results
                            "top_p": 0.9,
                            "repeat_penalty": 1.1
                        }
                    }
                )
                
                if response.status_code != 200:
                    raise Exception(f"Ollama API error: {response.status_code}")
                
                result = response.json()
                content = result.get("message", {}).get("content", "{}")
                
                try:
                    intent_data = json.loads(content)
                    
                    # Validate and enhance the result
                    intent_data = self._validate_and_enhance_intent(intent_data, text, context)
                    
                    self.logger.info(
                        f"Enhanced intent extracted: {intent_data['name']} "
                        f"(confidence: {intent_data['confidence']:.2f})"
                    )
                    return intent_data
                    
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse Ollama response: {content}")
                    return self._create_fallback_intent(text)
        
        except Exception as e:
            self.logger.error(f"Enhanced intent extraction failed: {e}")
            return self._create_fallback_intent(text)
    
    def _validate_and_enhance_intent(
        self, 
        intent_data: Dict[str, Any], 
        original_text: str, 
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Validate and enhance intent data"""
        # Set defaults
        intent_data.setdefault("name", "unknown")
        intent_data.setdefault("confidence", 0.5)
        intent_data.setdefault("entities", {})
        intent_data.setdefault("parameters", {})
        intent_data.setdefault("language", "en")
        intent_data.setdefault("reasoning", "")
        
        # Ensure confidence is within bounds
        intent_data["confidence"] = max(0.0, min(1.0, intent_data["confidence"]))
        
        # Enhance confidence based on context
        if context:
            # Higher confidence for longer, clearer audio
            if context.get("audio_duration", 0) > 2.0:
                intent_data["confidence"] *= 1.1
            
            # Higher confidence for speech with good voice segments
            if context.get("voice_segments", 0) > 0:
                intent_data["confidence"] *= 1.05
        
        # Text-based confidence adjustments
        text_lower = original_text.lower()
        
        # Boost confidence for clear command words
        command_words = {
            "schedule", "meeting", "appointment", "send", "email", "reminder", 
            "set", "create", "cancel", "search", "find", "call", "weather"
        }
        
        if any(word in text_lower for word in command_words):
            intent_data["confidence"] *= 1.1
        
        # Reduce confidence for very short or unclear text
        if len(original_text.split()) < 3:
            intent_data["confidence"] *= 0.8
        
        # Cap confidence
        intent_data["confidence"] = min(intent_data["confidence"], 1.0)
        
        return intent_data
    
    def _create_fallback_intent(self, text: str) -> Dict[str, Any]:
        """Create fallback intent when processing fails"""
        return {
            "name": "unknown",
            "confidence": 0.0,
            "entities": {},
            "parameters": {},
            "language": "en",
            "reasoning": "Intent extraction failed, using fallback"
        }
    
    async def _process_text_enhanced(
        self,
        text: str,
        context: Dict[str, Any] = None,
        user_id: Optional[str] = None,
        language: str = "en"
    ) -> Dict[str, Any]:
        """Enhanced text processing with context"""
        try:
            # Add user context if available
            enhanced_context = context or {}
            enhanced_context.update({
                "user_id": user_id,
                "input_language": language,
                "processing_method": "text_only"
            })
            
            # Extract intent with enhanced processing
            intent_result = await self._extract_intent_enhanced(text, enhanced_context)
            
            return {
                "text": text,
                "intent": intent_result,
                "language": language,
                "context": enhanced_context,
                "status": "success"
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced text processing failed: {e}")
            return {
                "text": text,
                "intent": self._create_fallback_intent(text),
                "status": "error",
                "error": str(e)
            }

# Create app instance
enhanced_voice_agent = EnhancedVoiceAgent()
app = enhanced_voice_agent.app

