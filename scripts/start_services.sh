# #!/bin/bash

# echo "Starting Voice-to-Action System..."

# # Start infrastructure services
# echo "Starting MongoDB and Redis..."
# docker-compose up -d mongodb redis

# # Wait for services to start
# echo "Waiting for services to start..."
# sleep 10

# # Start Ollama (if not already running)
# if ! pgrep -f "ollama serve" > /dev/null; then
#     echo "Starting Ollama..."
#     ollama serve &
#     sleep 5
# fi

# # Check if Mistral model is available
# if ! ollama list | grep -q "mistral:7b-instruct-v0.2"; then
#     echo "Pulling Mistral model..."
#     ollama pull mistral:7b-instruct-v0.2
# fi

# echo "✅ All services started successfully!"
