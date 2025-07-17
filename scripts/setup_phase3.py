#!/usr/bin/env python3
"""
Phase 3 Setup Script - Voice Processing Pipeline Setup and Verification
Complete setup verification for Phase 3 voice processing capabilities
"""

import asyncio
import sys
import os
import subprocess
import importlib.util
import tempfile
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


class Phase3Setup:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.setup_issues = []
        self.warnings = []

    def check_system_dependencies(self):
        """Check system-level dependencies"""
        print("🔧 Checking system dependencies...")

        # Check FFmpeg
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                print("   ✅ FFmpeg available")
            else:
                print("   ❌ FFmpeg not working properly")
                self.setup_issues.append("FFmpeg not working")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("   ❌ FFmpeg not found")
            self.setup_issues.append("FFmpeg not installed")
            print(
                "   💡 Install: sudo apt-get install ffmpeg (Ubuntu) or brew install ffmpeg (macOS)"
            )

        # Check audio system dependencies
        audio_deps = {
            "portaudio": "portaudio19-dev",
            "sndfile": "libsndfile1-dev",
            "ffi": "libffi-dev",
        }

        for dep, package in audio_deps.items():
            # We can't directly check these, but we'll check if related Python packages work
            print(f"   ℹ️ Audio dependency: {package} (install if audio issues occur)")

    def check_python_dependencies(self):
        """Check Python dependencies for Phase 3"""
        print("\n🐍 Checking Python dependencies...")

        # Core audio processing packages
        audio_packages = [
            ("whisper", "OpenAI Whisper"),
            ("torch", "PyTorch"),
            ("librosa", "Librosa audio processing"),
            ("soundfile", "SoundFile"),
            ("webrtcvad", "WebRTC VAD"),
            ("numpy", "NumPy"),
            ("scipy", "SciPy"),
        ]

        for package, description in audio_packages:
            try:
                importlib.import_module(package)
                print(f"   ✅ {description}")
            except ImportError:
                print(f"   ❌ {description} - Package '{package}' not found")
                self.setup_issues.append(f"Missing package: {package}")

        # Streamlit packages
        streamlit_packages = [
            ("streamlit", "Streamlit"),
            ("plotly", "Plotly"),
            ("pandas", "Pandas"),
        ]

        for package, description in streamlit_packages:
            try:
                importlib.import_module(package)
                print(f"   ✅ {description}")
            except ImportError:
                print(f"   ❌ {description} - Package '{package}' not found")
                self.setup_issues.append(f"Missing package: {package}")

        # Optional packages
        optional_packages = [
            ("streamlit_webrtc", "Streamlit WebRTC (for voice recording)")
        ]

        for package, description in optional_packages:
            try:
                importlib.import_module(package)
                print(f"   ✅ {description}")
            except ImportError:
                print(f"   ⚠️ {description} - Optional package '{package}' not found")
                self.warnings.append(f"Optional package missing: {package}")

    def check_gpu_availability(self):
        """Check GPU availability for faster processing"""
        print("\n🖥️ Checking GPU availability...")

        try:
            import torch

            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

                print(f"   ✅ CUDA GPU available: {gpu_name}")
                print(f"   📊 GPU memory: {gpu_memory:.1f} GB")
                print(f"   🔢 GPU count: {gpu_count}")

                if gpu_memory >= 6.0:
                    print("   ✅ Sufficient GPU memory for Whisper models")
                else:
                    print("   ⚠️ Low GPU memory - consider using smaller Whisper models")
                    self.warnings.append("Low GPU memory for optimal performance")
            else:
                print("   ⚠️ CUDA GPU not available - will use CPU")
                print("   💡 CPU processing will be slower but functional")
                self.warnings.append("No GPU available - CPU processing only")

        except ImportError:
            print("   ❌ PyTorch not available - cannot check GPU")
            self.setup_issues.append("PyTorch not installed")

    def check_whisper_models(self):
        """Check Whisper model availability"""
        print("\n🎤 Checking Whisper models...")

        try:
            import whisper

            # Check available models
            available_models = whisper.available_models()
            print(f"   📋 Available Whisper models: {', '.join(available_models)}")

            # Test loading base model
            try:
                print("   ⏳ Testing Whisper base model loading...")
                model = whisper.load_model("base")
                print("   ✅ Whisper base model loaded successfully")

                # Test with small audio
                import numpy as np

                test_audio = np.zeros(16000)  # 1 second of silence
                result = model.transcribe(test_audio)
                print("   ✅ Whisper transcription test successful")

            except Exception as e:
                print(f"   ❌ Whisper model test failed: {e}")
                self.setup_issues.append("Whisper model loading failed")

        except ImportError:
            print("   ❌ Whisper not available")
            self.setup_issues.append("Whisper not installed")

    def check_audio_processing(self):
        """Check audio processing capabilities"""
        print("\n🔊 Checking audio processing...")

        try:
            import librosa
            import soundfile as sf
            import numpy as np

            # Test basic audio processing
            print("   ⏳ Testing audio processing pipeline...")

            # Create test audio
            duration = 1.0
            sample_rate = 16000
            t = np.linspace(0, duration, int(sample_rate * duration))
            test_audio = 0.3 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

            # Test librosa functions
            audio_resampled = librosa.resample(
                test_audio, orig_sr=sample_rate, target_sr=22050
            )
            audio_mono = librosa.to_mono(test_audio)
            mfccs = librosa.feature.mfcc(y=test_audio, sr=sample_rate, n_mfcc=13)

            print("   ✅ Librosa audio processing working")

            # Test soundfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                sf.write(temp_file.name, test_audio, sample_rate)
                audio_loaded, sr_loaded = sf.read(temp_file.name)
                os.unlink(temp_file.name)

                if len(audio_loaded) > 0 and sr_loaded == sample_rate:
                    print("   ✅ SoundFile read/write working")
                else:
                    print("   ❌ SoundFile test failed")
                    self.setup_issues.append("SoundFile not working properly")

        except Exception as e:
            print(f"   ❌ Audio processing test failed: {e}")
            self.setup_issues.append("Audio processing libraries not working")

    def check_voice_activity_detection(self):
        """Check Voice Activity Detection"""
        print("\n🔊 Checking Voice Activity Detection...")

        try:
            import webrtcvad
            import numpy as np
            import struct

            # Test WebRTC VAD
            vad = webrtcvad.Vad()
            vad.set_mode(2)  # Aggressiveness level

            # Create test audio frame (30ms at 16kHz)
            frame_duration_ms = 30
            sample_rate = 16000
            frame_size = int(sample_rate * frame_duration_ms / 1000)

            # Generate test audio frame
            test_frame = np.random.randn(frame_size) * 0.1
            test_frame_int16 = (test_frame * 32767).astype(np.int16)
            frame_bytes = struct.pack(
                "<" + "h" * len(test_frame_int16), *test_frame_int16
            )

            # Test VAD
            is_speech = vad.is_speech(frame_bytes, sample_rate)
            print(f"   ✅ WebRTC VAD working (test result: {is_speech})")

        except ImportError:
            print("   ❌ WebRTC VAD not available")
            self.setup_issues.append("webrtcvad not installed")
        except Exception as e:
            print(f"   ❌ VAD test failed: {e}")
            self.setup_issues.append("VAD not working properly")

    async def check_infrastructure_services(self):
        """Check infrastructure services from Phase 2"""
        print("\n🔧 Checking infrastructure services...")

        # Check MongoDB
        try:
            import pymongo

            client = pymongo.MongoClient(
                "mongodb://localhost:27017/", serverSelectionTimeoutMS=5000
            )
            client.admin.command("ping")
            print("   ✅ MongoDB connection successful")

            # Check GridFS capability
            import gridfs

            db = client.ops_iq
            fs = gridfs.GridFS(db)
            print("   ✅ GridFS available for audio storage")

        except Exception as e:
            print(f"   ❌ MongoDB connection failed: {e}")
            self.setup_issues.append("MongoDB not running or not accessible")

        # Check Redis
        try:
            import redis

            r = redis.Redis(host="localhost", port=6379, db=0, socket_timeout=5)
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

                    # Check Mistral model
                    tags = response.json()
                    models = tags.get("models", [])
                    mistral_available = any(
                        "mistral" in model.get("name", "") for model in models
                    )

                    if mistral_available:
                        print("   ✅ Mistral model available")
                    else:
                        print("   ⚠️ Mistral model not found")
                        self.warnings.append("Mistral model not available")
                else:
                    print(f"   ❌ Ollama responded with status: {response.status_code}")
                    self.setup_issues.append("Ollama not responding correctly")
        except Exception as e:
            print(f"   ❌ Ollama connection failed: {e}")
            self.setup_issues.append("Ollama not running or not accessible")

    def check_phase3_files(self):
        """Check Phase 3 specific files"""
        print("\n📁 Checking Phase 3 files...")

        phase3_files = [
            ("services/whisper_service.py", "Whisper service"),
            ("services/audio_service.py", "Audio file management service"),
            ("services/voice_activity_detector.py", "Voice Activity Detection"),
            ("utils/audio_utils.py", "Audio processing utilities"),
            ("streamlit_app/voice_interface.py", "Streamlit voice interface"),
            ("scripts/test_phase3.py", "Phase 3 testing script"),
            ("requirements_phase3.txt", "Phase 3 requirements"),
        ]

        for file_path, description in phase3_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                print(f"   ✅ {description}")
            else:
                print(f"   ❌ {description} - Missing: {file_path}")
                self.setup_issues.append(f"Missing file: {file_path}")

    def check_streamlit_setup(self):
        """Check Streamlit interface setup"""
        print("\n🖥️ Checking Streamlit setup...")

        try:
            import streamlit as st

            print("   ✅ Streamlit available")

            # Check if streamlit app exists
            streamlit_app = self.project_root / "streamlit_app" / "voice_interface.py"
            if streamlit_app.exists():
                print("   ✅ Streamlit voice interface available")
                print(f"   🚀 Run with: streamlit run {streamlit_app}")
            else:
                print("   ❌ Streamlit voice interface not found")
                self.setup_issues.append("Streamlit voice interface missing")

        except ImportError:
            print("   ❌ Streamlit not available")
            self.setup_issues.append("Streamlit not installed")

    def check_agent_compatibility(self):
        """Check if Phase 2 agents are still working"""
        print("\n🤖 Checking agent compatibility...")

        try:
            # Check if we can import the enhanced voice agent
            sys.path.append(str(self.project_root))

            # Test imports
            from config.settings import settings

            print("   ✅ Settings configuration working")

            from config.logging import get_logger

            print("   ✅ Logging configuration working")

            # Check if base agent can be imported
            from agents.base_agent import BaseAgent

            print("   ✅ Base agent class available")

            # Check models
            from models.workflow import Workflow, WorkflowStatus, AgentType

            print("   ✅ Workflow models available")

            from models.responses import AgentResponse

            print("   ✅ Response models available")

        except ImportError as e:
            print(f"   ❌ Agent compatibility issue: {e}")
            self.setup_issues.append("Agent import problems")

    def check_database_compatibility(self):
        """Check database compatibility with Phase 3"""
        print("\n💾 Checking database compatibility...")

        try:
            import pymongo
            from motor.motor_asyncio import AsyncIOMotorClient
            import gridfs

            # Test MongoDB connection
            client = pymongo.MongoClient(
                "mongodb://localhost:27017/", serverSelectionTimeoutMS=5000
            )
            db = client.ops_iq

            # Test collections exist
            collections = db.list_collection_names()
            print(f"   📋 Existing collections: {collections}")

            # Test GridFS
            fs = gridfs.GridFS(db, collection="audio_files")
            print("   ✅ GridFS audio_files bucket available")

            # Test Beanie compatibility
            try:
                from beanie import init_beanie

                print("   ✅ Beanie ODM available")
            except ImportError:
                print("   ❌ Beanie ODM not available")
                self.setup_issues.append("Beanie ODM not installed")

        except Exception as e:
            print(f"   ❌ Database compatibility check failed: {e}")
            self.setup_issues.append("Database compatibility issues")

    def update_requirements(self):
        """Update requirements.txt with Phase 3 dependencies"""
        print("\n📦 Updating requirements.txt...")

        requirements_file = self.project_root / "requirements.txt"
        phase3_requirements_file = self.project_root / "requirements_phase3.txt"

        try:
            # Read Phase 3 requirements
            if phase3_requirements_file.exists():
                with open(phase3_requirements_file, "r") as f:
                    phase3_content = f.read()

                # Read existing requirements
                existing_content = ""
                if requirements_file.exists():
                    with open(requirements_file, "r") as f:
                        existing_content = f.read()

                # Check if Phase 3 requirements are already included
                if "openai-whisper" in existing_content:
                    print("   ✅ Phase 3 dependencies already in requirements.txt")
                else:
                    # Append Phase 3 requirements
                    with open(requirements_file, "a") as f:
                        f.write("\n# Phase 3 Voice Processing Dependencies\n")
                        # Extract just the package lines, not comments
                        lines = phase3_content.split("\n")
                        for line in lines:
                            line = line.strip()
                            if (
                                line
                                and not line.startswith("#")
                                and not line.startswith("==")
                            ):
                                f.write(line + "\n")
                    print("   ✅ Added Phase 3 dependencies to requirements.txt")
            else:
                print("   ⚠️ requirements_phase3.txt not found")
                self.warnings.append("Phase 3 requirements file missing")

        except Exception as e:
            print(f"   ❌ Failed to update requirements.txt: {e}")
            self.setup_issues.append("Failed to update requirements.txt")

    def create_missing_directories(self):
        """Create missing directories for Phase 3"""
        print("\n📁 Creating missing directories...")

        directories = ["streamlit_app", "services", "utils", "logs", "data/audio"]

        for directory in directories:
            dir_path = self.project_root / directory
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"   ✅ Created directory: {directory}")
            else:
                print(f"   ✅ Directory exists: {directory}")

    def test_whisper_installation(self):
        """Test Whisper installation specifically"""
        print("\n🎤 Testing Whisper installation...")

        try:
            import whisper
            import torch

            # Test model download location
            model_dir = whisper._get_model_path("base")
            print(f"   📁 Whisper models directory: {model_dir.parent}")

            # Check if we can load a small model quickly
            print("   ⏳ Testing tiny model loading...")
            tiny_model = whisper.load_model("tiny")
            print("   ✅ Whisper tiny model loaded successfully")

            # Test device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"   🖥️ Whisper will use device: {device}")

        except Exception as e:
            print(f"   ❌ Whisper installation test failed: {e}")
            self.setup_issues.append("Whisper installation problems")

    def generate_installation_commands(self):
        """Generate installation commands for missing dependencies"""
        print("\n💾 Installation Commands:")

        if self.setup_issues:
            print("   🔧 To fix critical issues:")

            if any("FFmpeg" in issue for issue in self.setup_issues):
                print("\n   1. Install FFmpeg:")
                print(
                    "      Ubuntu/Debian: sudo apt-get update && sudo apt-get install -y ffmpeg portaudio19-dev libsndfile1-dev libffi-dev"
                )
                print(
                    "      CentOS/RHEL: sudo yum install ffmpeg portaudio-devel libsndfile-devel libffi-devel"
                )
                print("      macOS: brew install ffmpeg portaudio libsndfile libffi")
                print(
                    "      Windows: Download FFmpeg from https://ffmpeg.org/ and add to PATH"
                )

            if any("package" in issue for issue in self.setup_issues):
                print("\n   2. Install Python packages:")
                print("      pip install -r requirements.txt")
                print("      # Or install Phase 3 specific packages:")
                print(
                    "      pip install openai-whisper torch librosa soundfile webrtcvad streamlit pandas plotly"
                )
                print("      # For GPU support (if you have CUDA):")
                print(
                    "      pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121"
                )

            if any("MongoDB" in issue for issue in self.setup_issues):
                print("\n   3. Start MongoDB:")
                print("      docker-compose up -d mongodb")
                print("      # Or install locally:")
                print("      # Ubuntu: sudo apt-get install mongodb")
                print("      # macOS: brew install mongodb-community")

            if any("Redis" in issue for issue in self.setup_issues):
                print("\n   4. Start Redis:")
                print("      docker-compose up -d redis")
                print("      # Or install locally:")
                print("      # Ubuntu: sudo apt-get install redis-server")
                print("      # macOS: brew install redis")

            if any("Ollama" in issue for issue in self.setup_issues):
                print("\n   5. Start Ollama:")
                print("      # Install Ollama:")
                print("      curl -fsSL https://ollama.ai/install.sh | sh")
                print("      # Start service:")
                print("      ollama serve &")
                print("      # Download Mistral model:")
                print("      ollama pull mistral:7b-instruct-v0.2")

        print("\n🚀 Phase 3 Complete Startup Commands:")
        print("   # 1. Install system dependencies")
        print(
            "   sudo apt-get install ffmpeg portaudio19-dev libsndfile1-dev  # Ubuntu"
        )
        print("   # brew install ffmpeg portaudio libsndfile  # macOS")
        print()
        print("   # 2. Install Python dependencies")
        print("   pip install -r requirements.txt")
        print()
        print("   # 3. Start infrastructure services")
        print("   docker-compose up -d mongodb redis")
        print("   ollama serve &")
        print()
        print("   # 4. Download Whisper models (first time)")
        print("   python -c \"import whisper; whisper.load_model('base')\"")
        print()
        print("   # 5. Start all agents")
        print("   python scripts/run_agents.py")
        print()
        print("   # 6. Test Phase 3 functionality")
        print("   python scripts/test_phase3.py")
        print()
        print("   # 7. Start Streamlit voice interface")
        print("   streamlit run streamlit_app/voice_interface.py")

    def print_summary(self):
        """Print setup summary"""
        print("\n" + "=" * 70)
        print("📋 PHASE 3 VOICE PROCESSING PIPELINE SETUP SUMMARY")
        print("=" * 70)

        if not self.setup_issues:
            print("🎉 Phase 3 setup complete! No critical issues found.")
            print("\n✅ Voice Processing Pipeline Ready:")
            print("   🎤 OpenAI Whisper speech-to-text integration")
            print("   📁 Audio file management with GridFS storage")
            print("   🔊 Voice Activity Detection with WebRTC VAD")
            print("   🛠️ Advanced audio processing utilities")
            print("   🖥️ Streamlit voice recording interface")
            print("   🤖 Enhanced voice agent with real NLP processing")
            print("   ⚡ GPU acceleration support (if available)")
            print("   🔄 End-to-end voice workflow pipeline")

            if self.warnings:
                print(f"\n⚠️ {len(self.warnings)} non-critical warnings:")
                for warning in self.warnings:
                    print(f"   • {warning}")
                print(
                    "\n💡 These warnings don't prevent Phase 3 from working but may affect performance."
                )

            print("\n🎯 What's New in Phase 3:")
            print("   • Real voice transcription (no more mocks!)")
            print("   • Audio file upload and storage")
            print("   • Voice activity detection")
            print("   • Enhanced intent recognition with context")
            print("   • Web-based voice recording interface")
            print("   • Performance monitoring and optimization")

            return True
        else:
            print(
                f"❌ Found {len(self.setup_issues)} critical issues that must be fixed:"
            )
            for i, issue in enumerate(self.setup_issues, 1):
                print(f"   {i}. {issue}")

            if self.warnings:
                print(f"\n⚠️ Also found {len(self.warnings)} warnings:")
                for warning in self.warnings:
                    print(f"   • {warning}")

            print("\n🔧 Phase 3 cannot start until critical issues are resolved.")
            print("💡 Run the installation commands above to fix these issues.")

            return False

    async def run_setup(self):
        """Run complete Phase 3 setup check"""
        print("🚀 Phase 3 Voice Processing Pipeline Setup and Verification")
        print("=" * 70)
        print("🎤 Checking real voice processing capabilities...")
        print()

        # Run all checks in order
        self.check_system_dependencies()
        self.check_python_dependencies()
        self.check_gpu_availability()
        self.check_whisper_models()
        self.check_audio_processing()
        self.check_voice_activity_detection()
        await self.check_infrastructure_services()
        self.check_phase3_files()
        self.check_streamlit_setup()
        self.check_agent_compatibility()
        self.check_database_compatibility()
        self.test_whisper_installation()

        # Setup improvements
        self.create_missing_directories()
        self.update_requirements()
        self.generate_installation_commands()

        # Print summary and return result
        return self.print_summary()


async def main():
    """Main setup function"""
    setup = Phase3Setup()
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
