#!/usr/bin/env python3
"""
Enhanced agent runner with better error handling and monitoring for Phase 2
"""
import subprocess
import sys
import time
import signal
import asyncio
from pathlib import Path
from datetime import datetime
import httpx

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

class AgentRunner:
    def __init__(self):
        self.processes = []
        self.agents = [
            ("master_agent", 8000),
            ("voice_agent", 8001), 
            ("action_agent", 8002)
        ]
        self.shutdown_requested = False
    
    def run_agent(self, agent_name, port):
        """Run a single agent"""
        cmd = [
            "uvicorn",
            f"agents.{agent_name}:app",
            "--host", "0.0.0.0",
            "--port", str(port),
            "--reload",
            "--log-level", "info"
        ]
        
        print(f"🚀 Starting {agent_name} on port {port}...")
        return subprocess.Popen(cmd)
    
    async def check_agent_health(self, port, max_retries=3):
        """Check if agent is healthy"""
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"http://localhost:{port}/health", timeout=5.0)
                    if response.status_code == 200:
                        return True
            except:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
        return False
    
    async def wait_for_agents(self):
        """Wait for all agents to be healthy"""
        print("⏳ Waiting for agents to start...")
        
        for agent_name, port in self.agents:
            print(f"   Checking {agent_name}...", end="")
            if await self.check_agent_health(port):
                print(" ✅")
            else:
                print(" ❌")
                return False
        
        return True
    
    async def monitor_agents(self):
        """Monitor agent health continuously"""
        while not self.shutdown_requested:
            try:
                # Check each agent's health
                for agent_name, port in self.agents:
                    if not await self.check_agent_health(port, max_retries=1):
                        print(f"⚠️  {agent_name} is not responding")
                
                # Check if any processes have died
                for i, process in enumerate(self.processes):
                    if process.poll() is not None:
                        agent_name, port = self.agents[i]
                        print(f"❌ {agent_name} process has died (exit code: {process.poll()})")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                print(f"⚠️ Monitoring error: {e}")
                await asyncio.sleep(30)
    
    def stop_all(self):
        """Stop all agent processes"""
        self.shutdown_requested = True
        
        for i, process in enumerate(self.processes):
            if process.poll() is None:  # Process is still running
                agent_name, _ = self.agents[i]
                print(f"🛑 Stopping {agent_name}...")
                process.terminate()
        
        # Wait for processes to terminate gracefully
        for process in self.processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    
    async def test_integration(self):
        """Test basic integration between agents"""
        print("\n🧪 Testing agent integration...")
        
        try:
            async with httpx.AsyncClient() as client:
                # Test creating a text workflow
                response = await client.post(
                    "http://localhost:8000/workflow/text",
                    json={
                        "text": "Schedule a meeting with John tomorrow at 2 PM",
                        "user_id": "test_user"
                    }
                )
                
                if response.status_code == 200:
                    workflow_data = response.json()
                    workflow_id = workflow_data.get("workflow_id")
                    print(f"✅ Text workflow created: {workflow_id}")
                    
                    # Wait a bit for processing
                    await asyncio.sleep(3)
                    
                    # Check workflow status
                    status_response = await client.get(f"http://localhost:8000/workflow/{workflow_id}")
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        print(f"✅ Workflow status: {status_data.get('status')}")
                        return True
                    else:
                        print(f"❌ Failed to get workflow status: {status_response.status_code}")
                else:
                    print(f"❌ Failed to create workflow: {response.status_code}")
                    
        except Exception as e:
            print(f"❌ Integration test failed: {e}")
        
        return False
    
    async def show_system_info(self):
        """Show system information and metrics"""
        print("\n📊 System Information:")
        
        try:
            async with httpx.AsyncClient() as client:
                # Get system metrics from master agent
                response = await client.get("http://localhost:8000/metrics")
                if response.status_code == 200:
                    metrics = response.json()
                    print(f"   Total workflows: {metrics.get('total_workflows', 0)}")
                    print(f"   Recent workflows (24h): {metrics.get('recent_workflows_24h', 0)}")
                    print(f"   Success rate (24h): {metrics.get('success_rate_24h', 0):.1%}")
                    print(f"   Active workflows: {metrics.get('active_workflows', 0)}")
                
                # Get health status from each agent
                print("\n🏥 Agent Health:")
                for agent_name, port in self.agents:
                    health_response = await client.get(f"http://localhost:{port}/health")
                    if health_response.status_code == 200:
                        health_data = health_response.json()
                        status = health_data.get("status", "unknown")
                        uptime = health_data.get("uptime", 0)
                        print(f"   {agent_name}: {status} (uptime: {uptime:.1f}s)")
                    else:
                        print(f"   {agent_name}: unhealthy")
                        
        except Exception as e:
            print(f"⚠️ Failed to get system info: {e}")
    
    async def run(self):
        """Run all agents with monitoring"""
        try:
            # Start all agents
            for agent_name, port in self.agents:
                process = self.run_agent(agent_name, port)
                self.processes.append(process)
                time.sleep(2)  # Stagger startup
            
            # Wait for agents to be ready
            if await self.wait_for_agents():
                print("\n🎉 All agents started successfully!")
                print("\n📚 API Documentation:")
                for agent_name, port in self.agents:
                    print(f"   {agent_name}: http://localhost:{port}/docs")
                
                print(f"\n⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Run integration test
                if await self.test_integration():
                    print("✅ Integration test passed!")
                else:
                    print("⚠️ Integration test failed - check logs")
                
                # Show system info
                await self.show_system_info()
                
                print("\n🔄 Press Ctrl+C to stop all agents...")
                print("📈 Monitoring agents (check every 30s)...")
                
                # Start monitoring
                await self.monitor_agents()
                
            else:
                print("❌ Some agents failed to start")
                self.stop_all()
                return 1
                
        except KeyboardInterrupt:
            print("\n🛑 Shutdown requested by user...")
            self.stop_all()
            print("✅ All agents stopped.")
            return 0
        except Exception as e:
            print(f"❌ Error: {e}")
            self.stop_all()
            return 1

def main():
    """Main entry point"""
    runner = AgentRunner()
    
    # Setup signal handlers
    def signal_handler(signum, frame):
        print(f"\n🛑 Received signal {signum}")
        runner.stop_all()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run the agent runner
    exit_code = asyncio.run(runner.run())
    sys.exit(exit_code)

if __name__ == "__main__":
    main()

