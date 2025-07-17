# Phase 2: Core Agent Development - Quick Start Guide

## 🎯 Phase 2 Overview

Phase 2 enhances the basic agent structure from Phase 1 with:
- **Complete data models** for workflows, commands, and responses
- **Enhanced agent communication** with workflow tracking
- **MongoDB integration** with Beanie ODM
- **Ollama/Mistral integration** for intent recognition
- **Comprehensive error handling** and monitoring
- **Full API documentation** with FastAPI

## 📋 Phase 2 Deliverables

### ✅ Data Models
- **Workflow tracking** with agent steps and metrics
- **Command structures** for different action types
- **Response models** for consistent API responses
- **MongoDB documents** with proper indexing

### ✅ Enhanced Agents
- **Master Agent**: Full workflow orchestration
- **Voice Agent**: Ollama/Mistral integration for NLP
- **Action Agent**: Mock external system integrations
- **Base Agent**: Common functionality and monitoring

### ✅ Communication Protocol
- **HTTP REST APIs** between agents
- **Redis message queues** for async communication
- **Workflow tracking** through MongoDB
- **Error handling** and retry mechanisms

## 🚀 Quick Start Commands

### 1. Verify Phase 2 Setup
```bash
python scripts/setup_phase2.py
```

### 2. Start Infrastructure Services
```bash
# Start MongoDB and Redis
docker-compose up -d mongodb redis

# Start Ollama (if not running)
ollama serve &

# Verify Mistral model
ollama pull mistral:7b-instruct-v0.2
```

### 3. Install/Update Dependencies
```bash
pip install -r requirements.txt
```

### 4. Start All Agents
```bash
python scripts/run_agents.py
```

### 5. Run Phase 2 Tests
```bash
python scripts/test_phase2.py
```

## 📚 API Documentation

Once agents are running, visit:
- **Master Agent**: http://localhost:8000/docs
- **Voice Agent**: http://localhost:8001/docs  
- **Action Agent**: http://localhost:8002/docs

## 🧪 Testing Phase 2

### Manual Testing Examples

#### 1. Create Text Workflow
```bash
curl -X POST "http://localhost:8000/workflow/text" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Schedule a meeting with John tomorrow at 2 PM",
    "user_id": "test_user"
  }'
```

#### 2. Create Voice Workflow (Mock)
```bash
curl -X POST "http://localhost:8000/workflow/voice" \
  -H "Content-Type: application/json" \
  -d '{
    "voice_input": {
      "transcription": "Send an email to Sarah about the project",
      "confidence": 0.95
    },
    "user_id": "test_user"
  }'
```

#### 3. Check Workflow Status
```bash
curl "http://localhost:8000/workflow/{workflow_id}"
```

#### 4. List Workflows
```bash
curl "http://localhost:8000/workflows?limit=10"
```

#### 5. Get System Metrics
```bash
curl "http://localhost:8000/metrics"
```

### Automated Testing
```bash
# Run comprehensive Phase 2 tests
python scripts/test_phase2.py

# Expected output: 15+ tests with high success rate
```

## 🔍 Key Features Demonstrated

### 1. **Workflow Orchestration**
- Master agent coordinates voice and action agents
- Complete workflow tracking from start to finish
- Progress monitoring and status updates

### 2. **Intent Recognition**  
- Ollama/Mistral integration for NLP
- Support for multiple intent types:
  - `schedule_meeting`
  - `send_email` 
  - `set_reminder`
  - `search_calendar`
  - `search_email`
  - `cancel_event`

### 3. **Mock Action Execution**
- Simulated external system integrations
- Realistic response times and data
- Error handling and status reporting

### 4. **Database Integration**
- MongoDB with Beanie ODM
- Proper indexing for performance
- Workflow persistence and querying

### 5. **Real-time Monitoring**
- Agent health checks
- Performance metrics collection
- Background monitoring tasks

## 📊 Expected Performance

### Response Times
- **Text workflow**: 2-5 seconds end-to-end
- **Voice workflow**: 3-6 seconds end-to-end  
- **Agent health checks**: < 1 second
- **Database queries**: < 500ms

### Throughput
- **Concurrent workflows**: 5-10 (single machine)
- **Requests per minute**: 50-100
- **Database operations**: 1000+ per minute

## 🔧 Troubleshooting

### Common Issues

#### 1. Agents Not Starting
```bash
# Check if ports are available
netstat -tulpn | grep :800

# Check logs
tail -f logs/app.log
```

#### 2. MongoDB Connection Failed
```bash
# Check MongoDB status
docker ps | grep mongodb

# Restart MongoDB
docker-compose restart mongodb
```

#### 3. Redis Connection Failed  
```bash
# Check Redis status
docker ps | grep redis

# Test Redis connection
redis-cli ping
```

#### 4. Ollama Not Responding
```bash
# Check Ollama status
ps aux | grep ollama

# Restart Ollama
killall ollama
ollama serve &

# Verify model
ollama list
```

#### 5. Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade

# Check Python path
python -c "import sys; print(sys.path)"
```

## 📈 Phase 2 Success Criteria

### ✅ Must Pass
- [ ] All agents start without errors
- [ ] MongoDB and Redis connections working
- [ ] Ollama/Mistral responding correctly
- [ ] Text workflows complete successfully
- [ ] Voice workflows (mock) complete successfully
- [ ] All health checks passing
- [ ] API documentation accessible

### ✅ Should Pass  
- [ ] 15+ automated tests passing
- [ ] Response times under 10 seconds
- [ ] Workflow listing and filtering working
- [ ] System metrics endpoint functional
- [ ] Error handling working correctly

### ✅ Nice to Have
- [ ] Integration test passing
- [ ] Performance metrics collection
- [ ] Background monitoring active
- [ ] Database indexes created

## 🎯 Next Steps to Phase 3

Once Phase 2 is complete:

1. **Verify all tests pass**: `python scripts/test_phase2.py`
2. **Check system metrics**: Ensure good performance
3. **Review logs**: No critical errors
4. **Test manual workflows**: Try different commands
5. **Document any issues**: Note for Phase 3

**Ready for Phase 3 when:**
- ✅ All core agent functionality working
- ✅ Workflow orchestration complete
- ✅ Database integration solid  
- ✅ Mock integrations functioning
- ✅ Monitoring and metrics active

## 📞 Support

If you encounter issues:

1. **Check the logs**: `tail -f logs/app.log`
2. **Run diagnostics**: `python scripts/setup_phase2.py`
3. **Test components**: `python scripts/test_phase2.py`
4. **Review documentation**: API docs at `/docs` endpoints
5. **Check dependencies**: Ensure all packages installed

---

**🎉 Phase 2 Complete!** 

Your multi-agent system now has:
- Full workflow orchestration
- Database persistence  
- Intent recognition
- Mock action execution
- Comprehensive monitoring

**Ready to move to Phase 3: Voice Processing Pipeline! 🎤**