#!/usr/bin/env python3
"""
Phase 2 Testing Script - Comprehensive testing of agent communication and workflows
"""
import asyncio
import httpx
import json
import sys
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings

class Phase2Tester:
    def __init__(self):
        self.base_urls = {
            "master": f"http://localhost:{settings.master_agent_port}",
            "voice": f"http://localhost:{settings.voice_agent_port}",
            "action": f"http://localhost:{settings.action_agent_port}"
        }
        self.test_results = []
        
    async def test_agent_health(self):
        """Test health endpoints for all agents"""
        print("🏥 Testing agent health endpoints...")
        
        async with httpx.AsyncClient() as client:
            for agent_name, base_url in self.base_urls.items():
                try:
                    response = await client.get(f"{base_url}/health", timeout=5.0)
                    if response.status_code == 200:
                        health_data = response.json()
                        status = health_data.get("status", "unknown")
                        print(f"   ✅ {agent_name}: {status}")
                        self.test_results.append(f"✅ {agent_name}_health")
                    else:
                        print(f"   ❌ {agent_name}: HTTP {response.status_code}")
                        self.test_results.append(f"❌ {agent_name}_health")
                except Exception as e:
                    print(f"   ❌ {agent_name}: {e}")
                    self.test_results.append(f"❌ {agent_name}_health")
    
    async def test_voice_agent_direct(self):
        """Test voice agent direct endpoints"""
        print("\n🎤 Testing voice agent direct endpoints...")
        
        async with httpx.AsyncClient() as client:
            # Test text processing
            try:
                response = await client.post(
                    f"{self.base_urls['voice']}/process_text",
                    json={"text": "Schedule a meeting with John tomorrow at 2 PM"},
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    intent = result.get("intent", {})
                    print(f"   ✅ Text processing: {intent.get('name', 'unknown')} (confidence: {intent.get('confidence', 0):.2f})")
                    self.test_results.append("✅ voice_text_processing")
                else:
                    print(f"   ❌ Text processing failed: HTTP {response.status_code}")
                    self.test_results.append("❌ voice_text_processing")
                    
            except Exception as e:
                print(f"   ❌ Text processing error: {e}")
                self.test_results.append("❌ voice_text_processing")
    
    async def test_action_agent_direct(self):
        """Test action agent direct endpoints"""
        print("\n⚡ Testing action agent direct endpoints...")
        
        async with httpx.AsyncClient() as client:
            # Test supported actions
            try:
                response = await client.get(f"{self.base_urls['action']}/supported_actions")
                if response.status_code == 200:
                    actions = response.json()
                    action_list = actions.get("actions", [])
                    print(f"   ✅ Supported actions: {len(action_list)} actions available")
                    self.test_results.append("✅ action_supported_actions")
                else:
                    print(f"   ❌ Supported actions failed: HTTP {response.status_code}")
                    self.test_results.append("❌ action_supported_actions")
            except Exception as e:
                print(f"   ❌ Supported actions error: {e}")
                self.test_results.append("❌ action_supported_actions")
            
            # Test direct action execution
            try:
                response = await client.post(
                    f"{self.base_urls['action']}/execute_action",
                    json={
                        "action_type": "schedule_meeting",
                        "parameters": {
                            "entities": {"person": "John", "subject": "Test Meeting"},
                            "parameters": {"duration": 60}
                        }
                    },
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"   ✅ Direct action execution: {result.get('status', 'unknown')}")
                    self.test_results.append("✅ action_direct_execution")
                else:
                    print(f"   ❌ Direct action execution failed: HTTP {response.status_code}")
                    self.test_results.append("❌ action_direct_execution")
                    
            except Exception as e:
                print(f"   ❌ Direct action execution error: {e}")
                self.test_results.append("❌ action_direct_execution")
    
    async def test_text_workflow_complete(self):
        """Test complete text workflow through master agent"""
        print("\n📝 Testing complete text workflow...")
        
        async with httpx.AsyncClient() as client:
            try:
                # Create text workflow
                workflow_response = await client.post(
                    f"{self.base_urls['master']}/workflow/text",
                    json={
                        "text": "Send an email to Sarah about the project update",
                        "user_id": "test_user_phase2",
                        "metadata": {"test": "phase2_text_workflow"}
                    },
                    timeout=10.0
                )
                
                if workflow_response.status_code == 200:
                    workflow_data = workflow_response.json()
                    workflow_id = workflow_data.get("workflow_id")
                    print(f"   ✅ Workflow created: {workflow_id}")
                    
                    # Wait for processing
                    print("   ⏳ Waiting for workflow processing...")
                    await asyncio.sleep(5)
                    
                    # Check workflow status
                    status_response = await client.get(
                        f"{self.base_urls['master']}/workflow/{workflow_id}",
                        timeout=5.0
                    )
                    
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        status = status_data.get("status")
                        progress = status_data.get("progress", 0)
                        print(f"   ✅ Workflow status: {status} (progress: {progress:.1%})")
                        
                        if status == "completed":
                            result = status_data.get("result", {})
                            if result.get("success"):
                                print("   ✅ Workflow completed successfully!")
                                self.test_results.append("✅ text_workflow_complete")
                            else:
                                print(f"   ⚠️ Workflow completed with issues: {result.get('message')}")
                                self.test_results.append("⚠️ text_workflow_complete")
                        else:
                            print(f"   ⚠️ Workflow not completed yet: {status}")
                            self.test_results.append("⚠️ text_workflow_incomplete")
                    else:
                        print(f"   ❌ Failed to get workflow status: HTTP {status_response.status_code}")
                        self.test_results.append("❌ text_workflow_status")
                else:
                    print(f"   ❌ Failed to create workflow: HTTP {workflow_response.status_code}")
                    self.test_results.append("❌ text_workflow_creation")
                    
            except Exception as e:
                print(f"   ❌ Text workflow error: {e}")
                self.test_results.append("❌ text_workflow_error")
    
    async def test_voice_workflow_mock(self):
        """Test voice workflow with mock voice input"""
        print("\n🎤 Testing voice workflow (mock)...")
        
        async with httpx.AsyncClient() as client:
            try:
                # Create voice workflow with mock data
                workflow_response = await client.post(
                    f"{self.base_urls['master']}/workflow/voice",
                    json={
                        "voice_input": {
                            "transcription": "Set a reminder for the dentist appointment next week",
                            "language": "en-US",
                            "confidence": 0.95,
                            "duration": 3.5
                        },
                        "user_id": "test_user_phase2",
                        "metadata": {"test": "phase2_voice_workflow"}
                    },
                    timeout=10.0
                )
                
                if workflow_response.status_code == 200:
                    workflow_data = workflow_response.json()
                    workflow_id = workflow_data.get("workflow_id")
                    print(f"   ✅ Voice workflow created: {workflow_id}")
                    
                    # Wait for processing
                    print("   ⏳ Waiting for voice workflow processing...")
                    await asyncio.sleep(5)
                    
                    # Check workflow status
                    status_response = await client.get(
                        f"{self.base_urls['master']}/workflow/{workflow_id}",
                        timeout=5.0
                    )
                    
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        status = status_data.get("status")
                        print(f"   ✅ Voice workflow status: {status}")
                        
                        if status == "completed":
                            print("   ✅ Voice workflow completed successfully!")
                            self.test_results.append("✅ voice_workflow_complete")
                        else:
                            print(f"   ⚠️ Voice workflow not completed: {status}")
                            self.test_results.append("⚠️ voice_workflow_incomplete")
                    else:
                        print(f"   ❌ Failed to get voice workflow status")
                        self.test_results.append("❌ voice_workflow_status")
                else:
                    print(f"   ❌ Failed to create voice workflow")
                    self.test_results.append("❌ voice_workflow_creation")
                    
            except Exception as e:
                print(f"   ❌ Voice workflow error: {e}")
                self.test_results.append("❌ voice_workflow_error")
    
    async def test_workflow_listing(self):
        """Test workflow listing and filtering"""
        print("\n📋 Testing workflow listing...")
        
        async with httpx.AsyncClient() as client:
            try:
                # List all workflows
                response = await client.get(
                    f"{self.base_urls['master']}/workflows?limit=10",
                    timeout=5.0
                )
                
                if response.status_code == 200:
                    workflows = response.json()
                    print(f"   ✅ Retrieved {len(workflows)} workflows")
                    self.test_results.append("✅ workflow_listing")
                    
                    # Test filtering by user
                    user_response = await client.get(
                        f"{self.base_urls['master']}/workflows?user_id=test_user_phase2&limit=5",
                        timeout=5.0
                    )
                    
                    if user_response.status_code == 200:
                        user_workflows = user_response.json()
                        print(f"   ✅ User filtering: {len(user_workflows)} workflows for test user")
                        self.test_results.append("✅ workflow_filtering")
                    else:
                        print("   ❌ User filtering failed")
                        self.test_results.append("❌ workflow_filtering")
                else:
                    print(f"   ❌ Workflow listing failed: HTTP {response.status_code}")
                    self.test_results.append("❌ workflow_listing")
                    
            except Exception as e:
                print(f"   ❌ Workflow listing error: {e}")
                self.test_results.append("❌ workflow_listing")
    
    async def test_system_metrics(self):
        """Test system metrics endpoint"""
        print("\n📊 Testing system metrics...")
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_urls['master']}/metrics",
                    timeout=5.0
                )
                
                if response.status_code == 200:
                    metrics = response.json()
                    total_workflows = metrics.get("total_workflows", 0)
                    success_rate = metrics.get("success_rate_24h", 0)
                    print(f"   ✅ Metrics retrieved: {total_workflows} total workflows, {success_rate:.1%} success rate")
                    self.test_results.append("✅ system_metrics")
                else:
                    print(f"   ❌ Metrics failed: HTTP {response.status_code}")
                    self.test_results.append("❌ system_metrics")
                    
            except Exception as e:
                print(f"   ❌ Metrics error: {e}")
                self.test_results.append("❌ system_metrics")
    
    async def test_error_handling(self):
        """Test error handling scenarios"""
        print("\n🚨 Testing error handling...")
        
        async with httpx.AsyncClient() as client:
            try:
                # Test invalid workflow ID
                response = await client.get(
                    f"{self.base_urls['master']}/workflow/invalid_id",
                    timeout=5.0
                )
                
                if response.status_code == 404:
                    print("   ✅ Invalid workflow ID correctly returns 404")
                    self.test_results.append("✅ error_invalid_workflow")
                else:
                    print(f"   ❌ Invalid workflow ID returned: {response.status_code}")
                    self.test_results.append("❌ error_invalid_workflow")
                
                # Test empty text input
                response = await client.post(
                    f"{self.base_urls['voice']}/process_text",
                    json={"text": ""},
                    timeout=5.0
                )
                
                if response.status_code == 400:
                    print("   ✅ Empty text input correctly returns 400")
                    self.test_results.append("✅ error_empty_text")
                else:
                    print(f"   ❌ Empty text input returned: {response.status_code}")
                    self.test_results.append("❌ error_empty_text")
                    
            except Exception as e:
                print(f"   ❌ Error handling test error: {e}")
                self.test_results.append("❌ error_handling")
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*60)
        print("📋 PHASE 2 TEST SUMMARY")
        print("="*60)
        
        passed = sum(1 for result in self.test_results if result.startswith("✅"))
        warned = sum(1 for result in self.test_results if result.startswith("⚠️"))
        failed = sum(1 for result in self.test_results if result.startswith("❌"))
        total = len(self.test_results)
        
        print(f"Total tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"⚠️ Warnings: {warned}")
        print(f"❌ Failed: {failed}")
        print(f"Success rate: {(passed / total * 100):.1f}%" if total > 0 else "No tests run")
        
        print("\n📝 Detailed Results:")
        for result in self.test_results:
            print(f"   {result}")
        
        # Determine overall status
        if failed == 0 and warned <= 2:
            print("\n🎉 Phase 2 testing PASSED! Ready to move to Phase 3.")
            return True
        elif failed <= 2:
            print("\n⚠️ Phase 2 testing completed with warnings. Review issues before Phase 3.")
            return True
        else:
            print("\n❌ Phase 2 testing FAILED. Fix critical issues before proceeding.")
            return False

async def main():
    """Main testing function"""
    print("🧪 Starting Phase 2 Comprehensive Testing")
    print("=" * 60)
    
    tester = Phase2Tester()
    
    # Run all tests
    await tester.test_agent_health()
    await tester.test_voice_agent_direct()
    await tester.test_action_agent_direct()
    await tester.test_text_workflow_complete()
    await tester.test_voice_workflow_mock()
    await tester.test_workflow_listing()
    await tester.test_system_metrics()
    await tester.test_error_handling()
    
    # Print summary and return status
    success = tester.print_summary()
    return 0 if success else 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Testing failed with error: {e}")
        sys.exit(1)

