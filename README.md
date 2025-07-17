# ops-iq - Multi-Agent Voice-to-Action System

## 🎯 Overview

A blend of "Ops" (operations) and "IQ", implies smart operations.
This is a sophisticated multi-agent system that converts voice commands into actionable tasks across various external services. The system uses local AI (Ollama + Mistral 7B) for intent recognition and integrates with external APIs.



## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Voice Agent   │    │  Master Agent   │    │  Action Agent   │
│   (Port 8001)   │◄──►│   (Port 8000)   │◄──►│   (Port 8002)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    Whisper      │    │   Workflow      │    │   External      │
│   (Speech-to-   │    │   Management    │    │   Services      │
│     Text)       │    │   (MongoDB)     │    │   (Google/MS)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Mistral 7B    │    │     Redis       │    │     Celery      │
│ (Intent & NER)  │    │   (Message      │    │  (Background    │
│                 │    │    Queue)       │    │    Tasks)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```


## 📋 Prerequisites

- **Python 3.9+**
- **Docker & Docker Compose** (Optional)
- **MongoDB** (for workflow storage)
- **Redis** (for message queue)
- **Ollama** (for local AI)

## 🛠️ Installation

### Clone Repository
```bash
git clone <repository-url>
cd ops-iq
```

### Set Up Python Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

### Multi-agents starts scripts
```bash

# start streamlit for ui interface
streamlit run streamlit_app/voice_interface.py

# start agents
python agent_master.py 
python agent_voice.py 
python agent_action.py 

```

### Install System Dependencies

#### Install Ollama
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows - Download from https://ollama.ai/download
```

#### Install Mistral 7B Model
```bash
ollama pull mistral:7b
```

#### Install Audio Processing Libraries
```bash
# macOS
brew install portaudio
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install portaudio19-dev python3-pyaudio
sudo apt-get install ffmpeg

# Windows
# Download and install from respective websites
```

### Set Up External Services

#### MongoDB Setup
```bash
# Using Docker (Recommended)
docker run -d --name mongodb -p 27017:27017 mongo:latest

# Or install locally
# macOS: brew install mongodb/brew/mongodb-community
# Ubuntu: sudo apt-get install mongodb
```

#### Redis Setup
```bash
# Using Docker (Recommended)
docker run -d --name redis -p 6379:6379 redis:latest

# Or install locally
# macOS: brew install redis
# Ubuntu: sudo apt-get install redis-server
```

### Configure Environment Variables

Create a `.env` file in the project root:

```env
# Database Configuration
MONGODB_URL=mongodb://localhost:27017
DATABASE_NAME=voice_actions
REDIS_URL=redis://localhost:6379

# Agent Ports
MASTER_AGENT_PORT=8000
VOICE_AGENT_PORT=8001
ACTION_AGENT_PORT=8002

# Ollama Configuration
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=mistral:7b

# Security
SECRET_KEY=your_secret_key_here
JWT_SECRET=your_jwt_secret_here

# Logging
LOG_LEVEL=INFO
```



#### Start Agents
```bash
# In terminal 2 - Master Agent
python -m agents.master_agent

# In terminal 3 - Voice Agent
python -m agents.voice_agent

# In terminal 4 - Action Agent
python -m agents.action_agent
```

#### Check Agent Health
```bash
# Master Agent
curl http://localhost:8000/health

# Voice Agent
curl http://localhost:8001/health

# Action Agent
curl http://localhost:8002/health
```

#### Test Voice Processing
```bash
# Test voice processing pipeline
python scripts/test_voice_processing.py
```

#### Test Action Execution
```bash
# Test external integrations
python scripts/test_external_actions.py
```

## 📊 Monitoring & Debugging

### Health Checks
```bash
# Check all services
curl http://localhost:8000/system/health

# Check specific service
curl http://localhost:8002/supported_actions
```


### Database Inspection
```bash
# MongoDB
mongosh
use voice_actions
db.workflows.find()

# Redis
redis-cli
keys *
```

## 🎤 Usage Examples

### 1. Schedule Meeting
```bash
# Voice command: "Schedule a meeting with John tomorrow at 2 PM"
curl -X POST http://localhost:8000/process_voice_command \
  -H "Content-Type: application/json" \
  -d '{
    "audio_file": "path/to/audio.wav",
    "user_id": "user123"
  }'
```

### 2. Send Email
```bash
# Voice command: "Send an email to Sarah about the project update"
curl -X POST http://localhost:8002/execute_action \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "send_email",
    "user_id": "user123",
    "parameters": {
      "to": ["sarah@example.com"],
      "subject": "Project Update",
      "body": "Here is the latest project update..."
    }
  }'
```

### 3. Search Calendar
```bash
# Voice command: "What meetings do I have today?"
curl -X POST http://localhost:8002/execute_action \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "search_calendar",
    "user_id": "user123",
    "parameters": {
      "query": "today"
    }
  }'
```

## 🔧 Configuration

### Audio Processing Settings
```python
# config/audio_settings.py
AUDIO_SETTINGS = {
    'sample_rate': 16000,
    'chunk_size': 1024,
    'channels': 1,
    'format': 'wav',
    'max_duration': 300,  # 5 minutes
    'noise_reduction': True
}
```

### AI Model Settings
```python
# config/ai_settings.py
AI_SETTINGS = {
    'model_name': 'mistral:7b',
    'temperature': 0.1,
    'max_tokens': 1000,
    'timeout': 30,
    'retry_attempts': 3
}
```


### Monitoring
- Response time targets: < 5 seconds
- Concurrent user support: 10+
- Uptime target: 99%+

## 🐛 Troubleshooting

### Common Issues

#### 1. Ollama Connection Error
```bash
# Check if Ollama is running
ollama list

# Restart Ollama
ollama serve
```

#### 2. MongoDB Connection Error
```bash
# Check MongoDB status
docker ps | grep mongo

# Restart MongoDB
docker restart mongodb
```

#### 3. Audio Processing Issues
```bash
# Check audio libraries
python -c "import pyaudio; print('PyAudio OK')"
python -c "import librosa; print('Librosa OK')"
```

#### 4. API Authentication Errors
- Verify credentials in `.env`
- Check API quotas in Google/Microsoft consoles
- Ensure redirect URIs match configuration

### Debug Mode
```bash
# Start with debug logging
export LOG_LEVEL=DEBUG
python -m agents.master_agent
```

## 📚 API Documentation

### Interactive API Docs
- Master Agent: http://localhost:8000/docs
- Voice Agent: http://localhost:8001/docs
- Action Agent: http://localhost:8002/docs

### OpenAPI Specs
- Master Agent: http://localhost:8000/openapi.json
- Voice Agent: http://localhost:8001/openapi.json
- Action Agent: http://localhost:8002/openapi.json

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Documentation**: `/docs` folder
- **Issues**: GitHub Issues
- **Discord**: Project Discord Server
- **Email**: support@voice-actions.com
