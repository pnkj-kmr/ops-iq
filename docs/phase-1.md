# Phase 1: Foundation Setup - Complete Guide
# Multi-Agent Voice-to-Action System

# ============================================================================
# STEP 1: SYSTEM REQUIREMENTS CHECK
# ============================================================================

# Check Python version (should be 3.9+, preferably 3.11)
python --version
python3 --version

# Check if pip is available
pip --version

# Check if Docker is installed
docker --version
docker-compose --version

# Check available disk space (need at least 20GB free)
df -h

# Check RAM (should have 16GB+)
free -h

# Check GPU (if available)
nvidia-smi

# ============================================================================
# STEP 2: PROJECT STRUCTURE CREATION
# ============================================================================

# Create main project directory
mkdir voice-to-action-system
cd voice-to-action-system

# Create project structure
mkdir -p agents
mkdir -p services  
mkdir -p models
mkdir -p config
mkdir -p utils
mkdir -p tests
mkdir -p docker
mkdir -p scripts
mkdir -p logs
mkdir -p data

# Create main files
touch README.md
touch requirements.txt
touch .env.example
touch .env
touch .gitignore
touch docker-compose.yml

# Create agent files
touch agents/__init__.py
touch agents/base_agent.py
touch agents/master_agent.py
touch agents/voice_agent.py
touch agents/action_agent.py

# Create service files
touch services/__init__.py
touch services/ollama_service.py
touch services/database_service.py
touch services/redis_service.py
touch services/audio_service.py

# Create model files
touch models/__init__.py
touch models/workflow.py
touch models/commands.py
touch models/responses.py

# Create config files
touch config/__init__.py
touch config/settings.py
touch config/database.py
touch config/logging.py

# Create utility files
touch utils/__init__.py
touch utils/audio_utils.py
touch utils/validation_utils.py
touch utils/logging_utils.py

# Create test files
touch tests/__init__.py
touch tests/test_agents.py
touch tests/test_services.py
touch tests/conftest.py

# Create Docker files
touch docker/Dockerfile.master
touch docker/Dockerfile.voice
touch docker/Dockerfile.action
touch docker/Dockerfile.streamlit

# Create scripts
touch scripts/setup.sh
touch scripts/run_agents.py
touch scripts/test_setup.py
touch scripts/start_services.sh

echo "✅ Project structure created successfully!"

# ============================================================================
# STEP 3: PYTHON VIRTUAL ENVIRONMENT SETUP
# ============================================================================

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

echo "✅ Virtual environment created and activated!"

# ============================================================================
# STEP 4: INSTALL PYTHON DEPENDENCIES
# ============================================================================

# Create requirements.txt with Phase 1 dependencies
cat > requirements.txt << 'EOF'
# Core Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-multipart==0.0.6

# Database
motor==3.3.2
beanie==1.23.6
redis==5.0.1
pymongo==4.6.0

# HTTP Client
httpx==0.25.2
requests==2.31.0

# Audio Processing
pyaudio==0.2.11
librosa==0.10.1
soundfile==0.12.1
numpy==1.24.3

# AI/ML
openai-whisper==20231117
torch>=2.0.0
transformers==4.35.0

# Utilities
python-dotenv==1.0.0
pyyaml==6.0.1
loguru==0.7.2
click==8.1.7

# Development
pytest==7.4.3
pytest-asyncio==0.21.1
black==23.11.0
flake8==6.1.0

# Streamlit (for Phase 5, but including now)
streamlit==1.28.1
streamlit-webrtc==0.47.1
plotly==5.17.0
pandas==2.1.3
EOF

# Install dependencies
pip install -r requirements.txt

echo "✅ Python dependencies installed!"

# ============================================================================
# STEP 5: OLLAMA INSTALLATION AND SETUP
# ============================================================================

# Install Ollama
echo "Installing Ollama..."
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve &

# Wait for Ollama to start
sleep 5

# Pull Mistral 7B model
echo "Downloading Mistral 7B model (this may take a while)..."
ollama pull mistral:7b-instruct-v0.2

# Test Ollama installation
echo "Testing Ollama installation..."
ollama run mistral:7b-instruct-v0.2 "Hello, are you working?"

echo "✅ Ollama and Mistral 7B installed and tested!"

# ============================================================================
# STEP 6: MONGODB SETUP
# ============================================================================

# Install MongoDB using Docker
echo "Setting up MongoDB..."
docker run --name mongodb -d -p 27017:27017 -v mongodb_data:/data/db mongo:7.0

# Wait for MongoDB to start
sleep 10

# Test MongoDB connection
python3 -c "
import pymongo
try:
    client = pymongo.MongoClient('mongodb://localhost:27017/')
    db = client.ops_iq
    collection = db.test_collection
    collection.insert_one({'test': 'Hello MongoDB'})
    print('✅ MongoDB connection successful!')
except Exception as e:
    print(f'❌ MongoDB connection failed: {e}')
"

# ============================================================================
# STEP 7: REDIS SETUP
# ============================================================================

# Install Redis using Docker
echo "Setting up Redis..."
docker run --name redis -d -p 6379:6379 redis:7.2-alpine

# Wait for Redis to start
sleep 5

# Test Redis connection
python3 -c "
import redis
try:
    r = redis.Redis(host='localhost', port=6379, db=0)
    r.set('test_key', 'Hello Redis')
    value = r.get('test_key')
    print(f'✅ Redis connection successful: {value.decode()}')
except Exception as e:
    print(f'❌ Redis connection failed: {e}')
"

# ============================================================================
# STEP 8: ENVIRONMENT CONFIGURATION
# ============================================================================

# Create .env file
cat > .env << 'EOF'
# Application Settings
APP_NAME=voice-to-action-system
APP_VERSION=0.1.0
DEBUG=true

# API Settings
MASTER_AGENT_HOST=localhost
MASTER_AGENT_PORT=8000
VOICE_AGENT_HOST=localhost
VOICE_AGENT_PORT=8001
ACTION_AGENT_HOST=localhost
ACTION_AGENT_PORT=8002

# Database Settings
MONGODB_URL=mongodb://localhost:27017/ops_iq
REDIS_URL=redis://localhost:6379

# Ollama Settings
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=mistral:7b-instruct-v0.2

# Audio Settings
AUDIO_SAMPLE_RATE=16000
AUDIO_CHUNK_SIZE=1024
AUDIO_CHANNELS=1

# Logging Settings
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# External API Keys (to be added later)
OPENAI_API_KEY=your-openai-key-here
GOOGLE_API_KEY=your-google-key-here
MICROSOFT_CLIENT_ID=your-microsoft-client-id-here
MICROSOFT_CLIENT_SECRET=your-microsoft-client-secret-here
EOF

echo "✅ Environment configuration created!"

# ============================================================================
# STEP 9: BASIC CONFIGURATION FILES
# ============================================================================

# Create settings.py
cat > config/settings.py << 'EOF'
from pydantic import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Application
    app_name: str = "voice-to-action-system"
    app_version: str = "0.1.0"
    debug: bool = True
    
    # API Settings
    master_agent_host: str = "localhost"
    master_agent_port: int = 8000
    voice_agent_host: str = "localhost"
    voice_agent_port: int = 8001
    action_agent_host: str = "localhost"
    action_agent_port: int = 8002
    
    # Database
    mongodb_url: str = "mongodb://localhost:27017/ops_iq"
    redis_url: str = "redis://localhost:6379"
    
    # Ollama
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "mistral:7b-instruct-v0.2"
    
    # Audio
    audio_sample_rate: int = 16000
    audio_chunk_size: int = 1024
    audio_channels: int = 1
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/app.log"
    
    # External APIs
    openai_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    
    class Config:
        env_file = ".env"

settings = Settings()
EOF

# Create database.py
cat > config/database.py << 'EOF'
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
import redis.asyncio as redis
from config.settings import settings

# MongoDB
class Database:
    client: AsyncIOMotorClient = None
    
database = Database()

async def connect_to_mongo():
    """Create database connection"""
    database.client = AsyncIOMotorClient(settings.mongodb_url)
    # We'll add document models in Phase 2
    print("✅ Connected to MongoDB")

async def close_mongo_connection():
    """Close database connection"""
    if database.client:
        database.client.close()
        print("✅ MongoDB connection closed")

# Redis
class RedisConnection:
    pool = None

redis_conn = RedisConnection()

async def connect_to_redis():
    """Create Redis connection"""
    redis_conn.pool = redis.ConnectionPool.from_url(settings.redis_url)
    print("✅ Connected to Redis")

async def close_redis_connection():
    """Close Redis connection"""
    if redis_conn.pool:
        await redis_conn.pool.disconnect()
        print("✅ Redis connection closed")
EOF

# Create logging configuration
cat > config/logging.py << 'EOF'
import logging
from loguru import logger
import sys
from config.settings import settings

# Configure loguru
logger.remove()
logger.add(sys.stderr, level=settings.log_level, format="{time} | {level} | {message}")
logger.add(settings.log_file, rotation="100 MB", retention="10 days", level=settings.log_level)

def get_logger(name: str):
    """Get logger instance"""
    return logger.bind(module=name)
EOF

echo "✅ Configuration files created!"

# ============================================================================
# STEP 10: BASIC AGENT STRUCTURE
# ============================================================================

# Create base agent class
cat > agents/base_agent.py << 'EOF'
from abc import ABC, abstractmethod
from typing import Dict, Any
import httpx
import json
from config.settings import settings
from config.logging import get_logger

class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(self, agent_name: str, port: int):
        self.agent_name = agent_name
        self.port = port
        self.logger = get_logger(agent_name)
        self.client = httpx.AsyncClient(timeout=30.0)
    
    @abstractmethod
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming request - implement in subclasses"""
        pass
    
    async def call_agent(self, agent_url: str, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP call to another agent"""
        try:
            response = await self.client.post(f"{agent_url}/{endpoint}", json=data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Agent call failed: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check endpoint"""
        return {
            "status": "healthy",
            "agent": self.agent_name,
            "port": self.port
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.client.aclose()
EOF

# Create master agent
cat > agents/master_agent.py << 'EOF'
from fastapi import FastAPI, HTTPException
from typing import Dict, Any
import uuid
from datetime import datetime
from agents.base_agent import BaseAgent
from config.settings import settings

class MasterAgent(BaseAgent):
    """Master agent for workflow orchestration"""
    
    def __init__(self):
        super().__init__("master_agent", settings.master_agent_port)
        self.app = FastAPI(title="Master Agent", version="0.1.0")
        self.workflows = {}  # In-memory storage for Phase 1
        self.setup_routes()
    
    def setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/health")
        async def health():
            return await self.health_check()
        
        @self.app.post("/process_command")
        async def process_command(request: Dict[str, Any]):
            return await self.process_request(request)
        
        @self.app.get("/workflow/{workflow_id}")
        async def get_workflow(workflow_id: str):
            if workflow_id in self.workflows:
                return self.workflows[workflow_id]
            raise HTTPException(status_code=404, detail="Workflow not found")
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming workflow request"""
        workflow_id = str(uuid.uuid4())
        
        # Create workflow record
        workflow = {
            "workflow_id": workflow_id,
            "status": "processing",
            "created_at": datetime.utcnow().isoformat(),
            "request": request
        }
        
        self.workflows[workflow_id] = workflow
        self.logger.info(f"Created workflow: {workflow_id}")
        
        try:
            # For Phase 1, just return success
            # In Phase 2, we'll add actual agent calls
            workflow["status"] = "completed"
            workflow["result"] = {
                "message": "Workflow processed successfully (Phase 1)",
                "workflow_id": workflow_id
            }
            
            return workflow
            
        except Exception as e:
            workflow["status"] = "failed"
            workflow["error"] = str(e)
            self.logger.error(f"Workflow {workflow_id} failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# Create app instance
master_agent = MasterAgent()
app = master_agent.app
EOF

# Create voice agent
cat > agents/voice_agent.py << 'EOF'
from fastapi import FastAPI
from typing import Dict, Any
from agents.base_agent import BaseAgent
from config.settings import settings

class VoiceAgent(BaseAgent):
    """Voice processing agent"""
    
    def __init__(self):
        super().__init__("voice_agent", settings.voice_agent_port)
        self.app = FastAPI(title="Voice Agent", version="0.1.0")
        self.setup_routes()
    
    def setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/health")
        async def health():
            return await self.health_check()
        
        @self.app.post("/process_audio")
        async def process_audio(request: Dict[str, Any]):
            return await self.process_request(request)
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process audio input"""
        self.logger.info("Processing audio request")
        
        # Phase 1: Mock response
        # In Phase 3, we'll add actual Whisper + Mistral integration
        return {
            "status": "success",
            "transcription": "Hello, this is a test transcription",
            "intent": "test_intent",
            "confidence": 0.95,
            "entities": {}
        }

# Create app instance
voice_agent = VoiceAgent()
app = voice_agent.app
EOF

# Create action agent
cat > agents/action_agent.py << 'EOF'
from fastapi import FastAPI
from typing import Dict, Any
from agents.base_agent import BaseAgent
from config.settings import settings

class ActionAgent(BaseAgent):
    """Action execution agent"""
    
    def __init__(self):
        super().__init__("action_agent", settings.action_agent_port)
        self.app = FastAPI(title="Action Agent", version="0.1.0")
        self.setup_routes()
    
    def setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/health")
        async def health():
            return await self.health_check()
        
        @self.app.post("/execute_action")
        async def execute_action(request: Dict[str, Any]):
            return await self.process_request(request)
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action"""
        self.logger.info("Executing action request")
        
        # Phase 1: Mock response
        # In Phase 4, we'll add actual external API integrations
        return {
            "status": "success",
            "action": "mock_action",
            "result": "Action executed successfully (Phase 1)",
            "execution_time": "1.2s"
        }

# Create app instance
action_agent = ActionAgent()
app = action_agent.app
EOF

echo "✅ Basic agent structure created!"

# ============================================================================
# STEP 11: DOCKER COMPOSE CONFIGURATION
# ============================================================================

# Create docker-compose.yml
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  mongodb:
    image: mongo:7.0
    container_name: mongodb
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db
    environment:
      MONGO_INITDB_DATABASE: ops_iq
    restart: unless-stopped

  redis:
    image: redis:7.2-alpine
    container_name: redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  ollama:
    image: ollama/ollama:latest
    container_name: ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    restart: unless-stopped
    # Note: You'll need to pull models manually after starting

volumes:
  mongodb_data:
  redis_data:
  ollama_data:
EOF

echo "✅ Docker Compose configuration created!"

# ============================================================================
# STEP 12: STARTUP SCRIPTS
# ============================================================================

# Create startup script
cat > scripts/start_services.sh << 'EOF'
#!/bin/bash

echo "Starting Voice-to-Action System..."

# Start infrastructure services
echo "Starting MongoDB and Redis..."
docker-compose up -d mongodb redis

# Wait for services to start
echo "Waiting for services to start..."
sleep 10

# Start Ollama (if not already running)
if ! pgrep -f "ollama serve" > /dev/null; then
    echo "Starting Ollama..."
    ollama serve &
    sleep 5
fi

# Check if Mistral model is available
if ! ollama list | grep -q "mistral:7b-instruct-v0.2"; then
    echo "Pulling Mistral model..."
    ollama pull mistral:7b-instruct-v0.2
fi

echo "✅ All services started successfully!"
EOF

# Create agent runner script
cat > scripts/run_agents.py << 'EOF'
#!/usr/bin/env python3
import subprocess
import sys
import time
import signal
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def run_agent(agent_name, port):
    """Run a single agent"""
    cmd = [
        "uvicorn",
        f"agents.{agent_name}:app",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--reload"
    ]
    
    print(f"Starting {agent_name} on port {port}...")
    return subprocess.Popen(cmd)

def main():
    """Run all agents"""
    processes = []
    
    try:
        # Start all agents
        processes.append(run_agent("master_agent", 8000))
        time.sleep(2)
        processes.append(run_agent("voice_agent", 8001))
        time.sleep(2)
        processes.append(run_agent("action_agent", 8002))
        
        print("✅ All agents started successfully!")
        print("📊 Master Agent:  http://localhost:8000/docs")
        print("🎤 Voice Agent:   http://localhost:8001/docs")
        print("⚡ Action Agent:  http://localhost:8002/docs")
        print("\nPress Ctrl+C to stop all agents...")
        
        # Wait for interrupt
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping all agents...")
        for process in processes:
            process.terminate()
        
        # Wait for processes to terminate
        for process in processes:
            process.wait()
        
        print("✅ All agents stopped.")

if __name__ == "__main__":
    main()
EOF

# Make scripts executable
chmod +x scripts/start_services.sh
chmod +x scripts/run_agents.py

echo "✅ Startup scripts created!"

# ============================================================================
# STEP 13: TESTING SCRIPT
# ============================================================================

# Create test script
cat > scripts/test_setup.py << 'EOF'
#!/usr/bin/env python3
import asyncio
import httpx
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings

async def test_services():
    """Test all services are running"""
    print("🧪 Testing system setup...")
    
    # Test MongoDB
    try:
        import pymongo
        client = pymongo.MongoClient(settings.mongodb_url)
        client.admin.command('ping')
        print("✅ MongoDB connection successful")
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        return False
    
    # Test Redis
    try:
        import redis
        r = redis.Redis.from_url(settings.redis_url)
        r.ping()
        print("✅ Redis connection successful")
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return False
    
    # Test Ollama
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{settings.ollama_url}/api/tags")
            if response.status_code == 200:
                print("✅ Ollama connection successful")
            else:
                print(f"❌ Ollama connection failed: {response.status_code}")
                return False
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        return False
    
    # Test Agents (if running)
    agents = [
        ("Master Agent", f"http://localhost:{settings.master_agent_port}/health"),
        ("Voice Agent", f"http://localhost:{settings.voice_agent_port}/health"),
        ("Action Agent", f"http://localhost:{settings.action_agent_port}/health")
    ]
    
    async with httpx.AsyncClient() as client:
        for name, url in agents:
            try:
                response = await client.get(url)
                if response.status_code == 200:
                    print(f"✅ {name} running successfully")
                else:
                    print(f"⚠️  {name} not responding (may not be started)")
            except Exception as e:
                print(f"⚠️  {name} not accessible: {e}")
    
    print("\n🎉 Setup testing complete!")
    return True

if __name__ == "__main__":
    asyncio.run(test_services())
EOF

echo "✅ Testing script created!"

# ============================================================================
# STEP 14: FINAL SETUP VERIFICATION
# ============================================================================

echo "🎯 Phase 1 setup complete! Running final verification..."

# Test Python imports
python3 -c "
import fastapi
import motor
import redis
import httpx
import whisper
print('✅ All Python dependencies imported successfully!')
"

# Test Ollama
if ollama list | grep -q "mistral"; then
    echo "✅ Mistral model available in Ollama"
else
    echo "⚠️  Mistral model not found. Run: ollama pull mistral:7b-instruct-v0.2"
fi

# Test Docker services
if docker ps | grep -q "mongodb"; then
    echo "✅ MongoDB container running"
else
    echo "⚠️  MongoDB container not running. Run: docker-compose up -d mongodb"
fi

if docker ps | grep -q "redis"; then
    echo "✅ Redis container running"
else
    echo "⚠️  Redis container not running. Run: docker-compose up -d redis"
fi

echo ""
echo "🚀 PHASE 1 SETUP COMPLETE!"
echo "=========================================="
echo "✅ Project structure created"
echo "✅ Python environment with dependencies"
echo "✅ Ollama + Mistral 7B installed"
echo "✅ MongoDB and Redis configured"
echo "✅ Basic agent structure created"
echo "✅ Docker configuration ready"
echo "✅ Startup scripts created"
echo ""
echo "🎯 NEXT STEPS:"
echo "1. Run: ./scripts/start_services.sh"
echo "2. Run: python scripts/run_agents.py"
echo "3. Test: python scripts/test_setup.py"
echo "4. Check: http://localhost:8000/docs"
echo ""
echo "🎉 Ready to move to Phase 2!"