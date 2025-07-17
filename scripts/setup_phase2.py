#!/usr/bin/env python3
"""
Phase 2 Setup Script - Verify Phase 1 and prepare for Phase 2
"""

import asyncio
import sys
import os
from pathlib import Path
import importlib.util

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


class Phase2Setup:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.setup_issues = []

    def check_file_exists(self, file_path: str, description: str) -> bool:
        """Check if a file exists"""
        full_path = self.project_root / file_path
        if full_path.exists():
            print(f"   ✅ {description}")
            return True
        else:
            print(f"   ❌ {description} - Missing: {file_path}")
            self.setup_issues.append(f"Missing file: {file_path}")
            return False

    def check_directory_structure(self):
        """Check if all required directories and files exist"""
        print("📁 Checking project structure...")

        # Directories
        directories = [
            ("agents", "Agents directory"),
            ("models", "Models directory"),
            ("config", "Config directory"),
            ("scripts", "Scripts directory"),
            ("tests", "Tests directory"),
            ("logs", "Logs directory"),
        ]

        for dir_name, description in directories:
            self.check_file_exists(dir_name, description)

        # Core files
        files = [
            ("requirements.txt", "Requirements file"),
            (".env", "Environment file"),
            ("docker-compose.yml", "Docker Compose file"),
            ("config/settings.py", "Settings configuration"),
            ("config/database.py", "Database configuration"),
            ("config/logging.py", "Logging configuration"),
        ]

        for file_path, description in files:
            self.check_file_exists(file_path, description)

    def check_phase2_files(self):
        """Check Phase 2 specific files"""
        print("\n📋 Checking Phase 2 files...")

        phase2_files = [
            ("models/workflow.py", "Workflow models"),
            ("models/commands.py", "Command models"),
            ("models/responses.py", "Response models"),
            ("agents/base_agent.py", "Enhanced base agent"),
            ("agents/master_agent.py", "Enhanced master agent"),
            ("agents/voice_agent.py", "Enhanced voice agent"),
            ("agents/action_agent.py", "Enhanced action agent"),
            ("scripts/run_agents.py", "Enhanced agent runner"),
            ("scripts/test_phase2.py", "Phase 2 testing script"),
        ]

        for file_path, description in phase2_files:
            self.check_file_exists(file_path, description)

    def check_python_imports(self):
        """Check if all required Python packages can be imported"""
        print("\n🐍 Checking Python dependencies...")

        required_packages = [
            ("fastapi", "FastAPI web framework"),
            ("uvicorn", "ASGI server"),
            ("pydantic", "Data validation"),
            ("motor", "Async MongoDB driver"),
            ("beanie", "MongoDB ODM"),
            ("redis", "Redis client"),
            ("httpx", "HTTP client"),
            ("pymongo", "MongoDB driver"),
        ]

        for package, description in required_packages:
            try:
                importlib.import_module(package)
                print(f"   ✅ {description}")
            except ImportError:
                print(f"   ❌ {description} - Package '{package}' not found")
                self.setup_issues.append(f"Missing package: {package}")

    async def check_infrastructure_services(self):
        """Check if infrastructure services are running"""
        print("\n🔧 Checking infrastructure services...")

        # Check MongoDB
        try:
            import pymongo

            client = pymongo.MongoClient(
                "mongodb://localhost:27017/", serverSelectionTimeoutMS=5000
            )
            client.admin.command("ping")
            print("   ✅ MongoDB connection successful")
        except Exception as e:
            print(f"   ❌ MongoDB connection failed: {e}")
            self.setup_issues.append("MongoDB not running or not accessible")

        # Check Redis
        try:
            import redis

            r = redis.Redis(
                host="localhost", port=6379, db=0, socket_timeout=5, password="Infra0n"
            )
            r.ping()
            print("   ✅ Redis connection successful")
        except Exception as e:
            print(f"   ❌ Redis connection failed: {e}")
            self.setup_issues.append("Redis not running or not accessible")

        # Check Ollama
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "http://localhost:11434/api/tags", timeout=5.0
                )
                if response.status_code == 200:
                    print("   ✅ Ollama connection successful")

                    # Check if Mistral model is available
                    tags = response.json()
                    models = tags.get("models", [])
                    mistral_available = any(
                        "mistral" in model.get("name", "") for model in models
                    )

                    if mistral_available:
                        print("   ✅ Mistral model available")
                    else:
                        print(
                            "   ⚠️ Mistral model not found - run: ollama pull mistral:7b-instruct-v0.2"
                        )
                        self.setup_issues.append("Mistral model not available")
                else:
                    print(f"   ❌ Ollama responded with status: {response.status_code}")
                    self.setup_issues.append("Ollama not responding correctly")
        except Exception as e:
            print(f"   ❌ Ollama connection failed: {e}")
            self.setup_issues.append("Ollama not running or not accessible")

    def check_environment_variables(self):
        """Check if required environment variables are set"""
        print("\n🌍 Checking environment variables...")

        try:
            from config.settings import settings

            # Check critical settings
            if settings.mongodb_url:
                print("   ✅ MongoDB URL configured")
            else:
                print("   ❌ MongoDB URL not configured")
                self.setup_issues.append("MongoDB URL not set")

            if settings.redis_url:
                print("   ✅ Redis URL configured")
            else:
                print("   ❌ Redis URL not configured")
                self.setup_issues.append("Redis URL not set")

            if settings.ollama_url:
                print("   ✅ Ollama URL configured")
            else:
                print("   ❌ Ollama URL not configured")
                self.setup_issues.append("Ollama URL not set")

            # Check ports
            if all(
                [
                    settings.master_agent_port,
                    settings.voice_agent_port,
                    settings.action_agent_port,
                ]
            ):
                print("   ✅ Agent ports configured")
            else:
                print("   ❌ Agent ports not properly configured")
                self.setup_issues.append("Agent ports not configured")

        except Exception as e:
            print(f"   ❌ Settings configuration error: {e}")
            self.setup_issues.append("Settings configuration failed")

    def create_missing_directories(self):
        """Create any missing directories"""
        print("\n📁 Creating missing directories...")

        directories = ["logs", "data", "tests", "scripts"]

        for directory in directories:
            dir_path = self.project_root / directory
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"   ✅ Created directory: {directory}")
            else:
                print(f"   ✅ Directory exists: {directory}")

    def update_requirements(self):
        """Update requirements.txt with Phase 2 dependencies"""
        print("\n📦 Updating requirements.txt...")

        phase2_requirements = """
# Phase 2 additional dependencies
beanie>=1.23.6
redis>=5.0.1
httpx>=0.25.2
"""

        requirements_file = self.project_root / "requirements.txt"

        try:
            # Read existing requirements
            if requirements_file.exists():
                with open(requirements_file, "r") as f:
                    existing_content = f.read()

                # Check if Phase 2 dependencies are already there
                if "beanie" in existing_content:
                    print("   ✅ Phase 2 dependencies already in requirements.txt")
                else:
                    # Append Phase 2 requirements
                    with open(requirements_file, "a") as f:
                        f.write(phase2_requirements)
                    print("   ✅ Added Phase 2 dependencies to requirements.txt")
            else:
                print("   ❌ requirements.txt not found")
                self.setup_issues.append("requirements.txt not found")

        except Exception as e:
            print(f"   ❌ Failed to update requirements.txt: {e}")
            self.setup_issues.append("Failed to update requirements.txt")

    def generate_startup_commands(self):
        """Generate helpful startup commands"""
        print("\n🚀 Startup Commands:")
        print("   1. Start infrastructure services:")
        print("      docker-compose up -d mongodb redis")
        print("      ollama serve &")
        print("")
        print("   2. Install/update dependencies:")
        print("      pip install -r requirements.txt")
        print("")
        print("   3. Start all agents:")
        print("      python scripts/run_agents.py")
        print("")
        print("   4. Run Phase 2 tests:")
        print("      python scripts/test_phase2.py")
        print("")
        print("   5. Check API documentation:")
        print("      http://localhost:8000/docs (Master Agent)")
        print("      http://localhost:8001/docs (Voice Agent)")
        print("      http://localhost:8002/docs (Action Agent)")

    def print_summary(self):
        """Print setup summary"""
        print("\n" + "=" * 60)
        print("📋 PHASE 2 SETUP SUMMARY")
        print("=" * 60)

        if not self.setup_issues:
            print("🎉 Phase 2 setup complete! No issues found.")
            print("\n✅ Ready to start Phase 2 development:")
            print("   1. All files and directories are in place")
            print("   2. All dependencies are available")
            print("   3. Infrastructure services are running")
            print("   4. Configuration is correct")
            return True
        else:
            print(f"⚠️ Found {len(self.setup_issues)} setup issues:")
            for i, issue in enumerate(self.setup_issues, 1):
                print(f"   {i}. {issue}")

            print("\n🔧 Please fix these issues before starting Phase 2:")
            if "MongoDB" in str(self.setup_issues):
                print("   - Start MongoDB: docker-compose up -d mongodb")
            if "Redis" in str(self.setup_issues):
                print("   - Start Redis: docker-compose up -d redis")
            if "Ollama" in str(self.setup_issues):
                print("   - Start Ollama: ollama serve")
            if "package" in str(self.setup_issues):
                print("   - Install packages: pip install -r requirements.txt")

            return False

    async def run_setup(self):
        """Run complete Phase 2 setup check"""
        print("🚀 Phase 2 Setup and Verification")
        print("=" * 60)

        # Run all checks
        self.check_directory_structure()
        self.check_phase2_files()
        self.check_python_imports()
        await self.check_infrastructure_services()
        self.check_environment_variables()

        # Fixes and improvements
        self.create_missing_directories()
        self.update_requirements()

        # Generate helpful commands
        self.generate_startup_commands()

        # Print summary
        return self.print_summary()


async def main():
    """Main setup function"""
    setup = Phase2Setup()
    success = await setup.run_setup()
    return 0 if success else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        sys.exit(1)
