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
        client.admin.command("ping")
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
            print(f"---- settings.ollama_url -- {settings.ollama_url}")
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
        ("Action Agent", f"http://localhost:{settings.action_agent_port}/health"),
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
