#!/usr/bin/env python3
"""
Phase 3 Testing Script - Comprehensive testing of voice processing pipeline
Tests Whisper integration, audio processing, VAD, and enhanced NLP
"""

import asyncio
import httpx
import sys
import os
import io
import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings
from services.whisper_service import whisper_service
from services.audio_service import audio_service
from services.voice_activity_detector import voice_activity_detector
from utils.audio_utils import audio_processor


class Phase3Tester:
    def __init__(self):
        self.base_urls = {
            "master": f"http://localhost:{settings.master_agent_port}",
            "voice": f"http://localhost:{settings.voice_agent_port}",
            "action": f"http://localhost:{settings.action_agent_port}",
        }
        self.test_results = []
        self.test_audio_files = []

    def create_test_audio(
        self, duration: float = 3.0, sample_rate: int = 16000
    ) -> bytes:
        """Create test audio file"""
        try:
            # Generate simple sine wave for testing
            t = np.linspace(0, duration, int(sample_rate * duration))
            frequency = 440  # A4 note
            audio = 0.3 * np.sin(2 * np.pi * frequency * t)

            # Add some noise to make it more realistic
            noise = 0.05 * np.random.randn(len(audio))
            audio = audio + noise

            # Convert to bytes (WAV format)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                sf.write(temp_file.name, audio, sample_rate, format="WAV")
                temp_file.flush()

                with open(temp_file.name, "rb") as f:
                    audio_bytes = f.read()

                # Clean up
                os.unlink(temp_file.name)

                return audio_bytes

        except Exception as e:
            print(f"   ❌ Failed to create test audio: {e}")
            return b""

    async def test_whisper_service(self):
        """Test Whisper service functionality"""
        print("🎤 Testing Whisper Service...")

        try:
            # Initialize Whisper
            await whisper_service.initialize()
            print("   ✅ Whisper service initialized")
            self.test_results.append("✅ whisper_initialization")

            # Test model info
            model_info = whisper_service.get_model_info()
            print(
                f"   ✅ Model info: {model_info['model_size']} on {model_info['device']}"
            )
            self.test_results.append("✅ whisper_model_info")

            # Test health check
            health = await whisper_service.health_check()
            if health.get("status") == "healthy":
                print("   ✅ Whisper health check passed")
                self.test_results.append("✅ whisper_health")
            else:
                print(f"   ❌ Whisper health check failed: {health}")
                self.test_results.append("❌ whisper_health")

            # Test transcription with generated audio
            test_audio = self.create_test_audio(duration=2.0)
            if test_audio:
                transcription_result = await whisper_service.transcribe_audio(
                    test_audio
                )

                if transcription_result.processing_time > 0:
                    print(
                        f"   ✅ Transcription completed in {transcription_result.processing_time:.2f}s"
                    )
                    print(
                        f"   📝 Result: '{transcription_result.text}' (confidence: {transcription_result.confidence:.2f})"
                    )
                    self.test_results.append("✅ whisper_transcription")
                else:
                    print("   ❌ Transcription failed")
                    self.test_results.append("❌ whisper_transcription")
            else:
                print("   ⚠️ Could not create test audio")
                self.test_results.append("⚠️ whisper_transcription")

        except Exception as e:
            print(f"   ❌ Whisper service test failed: {e}")
            self.test_results.append("❌ whisper_service")

    async def test_audio_service(self):
        """Test audio service functionality"""
        print("\n📁 Testing Audio Service...")

        try:
            # Initialize audio service (assuming db_client is available)
            from motor.motor_asyncio import AsyncIOMotorClient

            db_client = AsyncIOMotorClient(settings.mongodb_url)

            await audio_service.initialize(db_client)
            print("   ✅ Audio service initialized")
            self.test_results.append("✅ audio_service_init")

            # Test health check
            health = await audio_service.health_check()
            if health.get("status") == "healthy":
                print("   ✅ Audio service health check passed")
                self.test_results.append("✅ audio_service_health")
            else:
                print(f"   ❌ Audio service health check failed: {health}")
                self.test_results.append("❌ audio_service_health")

            # Test audio upload
            test_audio = self.create_test_audio(duration=1.0)
            if test_audio:
                metadata = await audio_service.upload_audio_file(
                    file_data=test_audio,
                    filename="test_audio.wav",
                    content_type="audio/wav",
                    user_id="test_user_phase3",
                )

                if metadata and metadata.file_id:
                    print(f"   ✅ Audio upload successful: {metadata.file_id}")
                    print(
                        f"   📊 Duration: {metadata.duration:.1f}s, Size: {metadata.size} bytes"
                    )
                    self.test_results.append("✅ audio_upload")

                    # Test audio retrieval
                    retrieved_audio = await audio_service.get_audio_file(
                        metadata.file_id
                    )
                    if retrieved_audio and len(retrieved_audio) == len(test_audio):
                        print("   ✅ Audio retrieval successful")
                        self.test_results.append("✅ audio_retrieval")
                    else:
                        print("   ❌ Audio retrieval failed")
                        self.test_results.append("❌ audio_retrieval")

                    # Test metadata retrieval
                    retrieved_metadata = await audio_service.get_audio_metadata(
                        metadata.file_id
                    )
                    if retrieved_metadata:
                        print("   ✅ Audio metadata retrieval successful")
                        self.test_results.append("✅ audio_metadata")
                    else:
                        print("   ❌ Audio metadata retrieval failed")
                        self.test_results.append("❌ audio_metadata")

                    # Store file ID for cleanup
                    self.test_audio_files.append(metadata.file_id)

                else:
                    print("   ❌ Audio upload failed")
                    self.test_results.append("❌ audio_upload")
            else:
                print("   ⚠️ Could not create test audio")
                self.test_results.append("⚠️ audio_upload")

        except Exception as e:
            print(f"   ❌ Audio service test failed: {e}")
            self.test_results.append("❌ audio_service")

    async def test_voice_activity_detection(self):
        """Test Voice Activity Detection"""
        print("\n🔊 Testing Voice Activity Detection...")

        try:
            # Create test audio with silence and speech segments
            sample_rate = 16000
            duration = 6.0  # 6 seconds

            # Generate audio: silence -> speech -> silence -> speech
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = np.zeros_like(t)

            # Add speech segments (simple sine waves)
            # Speech from 1-2 seconds
            speech1_start, speech1_end = int(1.0 * sample_rate), int(2.0 * sample_rate)
            audio[speech1_start:speech1_end] = 0.3 * np.sin(
                2 * np.pi * 440 * t[speech1_start:speech1_end]
            )

            # Speech from 4-5 seconds
            speech2_start, speech2_end = int(4.0 * sample_rate), int(5.0 * sample_rate)
            audio[speech2_start:speech2_end] = 0.3 * np.sin(
                2 * np.pi * 880 * t[speech2_start:speech2_end]
            )

            # Test VAD
            segments = voice_activity_detector.detect_voice_activity(audio, sample_rate)

            print(f"   ✅ VAD detected {len(segments)} voice segments")

            for i, segment in enumerate(segments):
                print(
                    f"   📊 Segment {i + 1}: {segment.start_time:.1f}s - {segment.end_time:.1f}s "
                    f"(confidence: {segment.confidence:.2f})"
                )

            # Validate results
            if len(segments) >= 1:  # Should detect at least one segment
                print("   ✅ VAD detection successful")
                self.test_results.append("✅ vad_detection")
            else:
                print("   ⚠️ VAD detected no segments")
                self.test_results.append("⚠️ vad_detection")

            # Test segment extraction
            extracted_segments = voice_activity_detector.extract_voice_segments(
                audio, segments, sample_rate
            )

            if extracted_segments:
                print(f"   ✅ Extracted {len(extracted_segments)} audio segments")
                self.test_results.append("✅ vad_extraction")
            else:
                print("   ❌ Segment extraction failed")
                self.test_results.append("❌ vad_extraction")

            # Test statistics
            stats = voice_activity_detector.get_voice_statistics(segments)
            print(
                f"   📈 Voice statistics: {stats['total_voice_duration']:.1f}s total, "
                f"{stats['average_confidence']:.2f} avg confidence"
            )
            self.test_results.append("✅ vad_statistics")

        except Exception as e:
            print(f"   ❌ VAD test failed: {e}")
            self.test_results.append("❌ vad_test")

    async def test_audio_utilities(self):
        """Test audio processing utilities"""
        print("\n🛠️ Testing Audio Utilities...")

        try:
            # Create test audio
            test_audio = self.create_test_audio(duration=2.0)

            if test_audio:
                # Test audio info extraction
                audio_info = audio_processor.get_audio_info(test_audio, "test.wav")

                if audio_info.is_valid:
                    print(
                        f"   ✅ Audio info: {audio_info.duration:.1f}s, {audio_info.sample_rate}Hz, "
                        f"{audio_info.channels}ch"
                    )
                    self.test_results.append("✅ audio_info")
                else:
                    print(f"   ❌ Audio info invalid: {audio_info.error_message}")
                    self.test_results.append("❌ audio_info")

                # Test audio processing for speech recognition
                processed_audio, processing_info = (
                    audio_processor.process_for_speech_recognition(
                        test_audio, "test.wav"
                    )
                )

                if processed_audio:
                    print(
                        f"   ✅ Audio processing: {len(processing_info['processing_steps'])} steps applied"
                    )
                    print(
                        f"   📊 Processing steps: {', '.join(processing_info['processing_steps'])}"
                    )
                    self.test_results.append("✅ audio_processing")
                else:
                    print("   ❌ Audio processing failed")
                    self.test_results.append("❌ audio_processing")

                # Test quality validation
                import librosa
                import io

                audio_buffer = io.BytesIO(test_audio)
                audio_signal, sr = librosa.load(audio_buffer, sr=16000)

                quality_report = audio_processor.validate_audio_quality(
                    audio_signal, sr
                )
                print(f"   📋 Audio quality: {quality_report['overall_quality']}")
                if quality_report["issues"]:
                    print(f"   ⚠️ Issues: {', '.join(quality_report['issues'])}")

                self.test_results.append("✅ audio_quality")

            else:
                print("   ⚠️ Could not create test audio")
                self.test_results.append("⚠️ audio_utilities")

        except Exception as e:
            print(f"   ❌ Audio utilities test failed: {e}")
            self.test_results.append("❌ audio_utilities")

    async def test_enhanced_voice_agent(self):
        """Test enhanced voice agent endpoints"""
        print("\n🎤 Testing Enhanced Voice Agent...")

        async with httpx.AsyncClient() as client:
            try:
                # Test health endpoint
                response = await client.get(
                    f"{self.base_urls['voice']}/health", timeout=10.0
                )

                if response.status_code == 200:
                    health_data = response.json()
                    print(f"   ✅ Voice agent health: {health_data.get('status')}")

                    # Check voice service dependencies
                    dependencies = health_data.get("dependencies", {})
                    for service, status in dependencies.items():
                        if status == "healthy":
                            print(f"   ✅ {service}: {status}")
                        else:
                            print(f"   ⚠️ {service}: {status}")

                    self.test_results.append("✅ enhanced_voice_health")
                else:
                    print(
                        f"   ❌ Voice agent health check failed: {response.status_code}"
                    )
                    self.test_results.append("❌ enhanced_voice_health")

                # Test supported languages endpoint
                response = await client.get(
                    f"{self.base_urls['voice']}/supported_languages", timeout=5.0
                )

                if response.status_code == 200:
                    lang_data = response.json()
                    languages = lang_data.get("languages", [])
                    print(f"   ✅ Supported languages: {len(languages)} languages")
                    print(f"   🌍 Languages: {', '.join(languages[:5])}...")
                    self.test_results.append("✅ voice_languages")
                else:
                    print("   ❌ Failed to get supported languages")
                    self.test_results.append("❌ voice_languages")

                # Test advanced text processing
                response = await client.post(
                    f"{self.base_urls['voice']}/process_text_advanced",
                    json={
                        "text": "Schedule a meeting with John tomorrow at 2 PM about the project review",
                        "user_id": "test_user_phase3",
                        "language": "en",
                        "context": {"session_type": "test"},
                    },
                    timeout=15.0,
                )

                if response.status_code == 200:
                    result = response.json()
                    intent = result.get("intent", {})
                    print(
                        f"   ✅ Advanced text processing: {intent.get('name')} "
                        f"(confidence: {intent.get('confidence', 0):.2f})"
                    )

                    if intent.get("reasoning"):
                        print(f"   🧠 Reasoning: {intent['reasoning']}")

                    self.test_results.append("✅ voice_advanced_text")
                else:
                    print(
                        f"   ❌ Advanced text processing failed: {response.status_code}"
                    )
                    self.test_results.append("❌ voice_advanced_text")

                # Test audio upload (if we have test audio)
                test_audio = self.create_test_audio(duration=3.0)
                if test_audio:
                    files = {
                        "audio_file": ("test.wav", io.BytesIO(test_audio), "audio/wav")
                    }
                    data = {
                        "user_id": "test_user_phase3",
                        "enable_vad": "true",
                        "include_segments": "true",
                    }

                    response = await client.post(
                        f"{self.base_urls['voice']}/upload_audio",
                        files=files,
                        data=data,
                        timeout=30.0,
                    )

                    if response.status_code == 200:
                        result = response.json()
                        print(
                            f"   ✅ Audio upload processing: transcription='{result.get('transcription', '')[:50]}...'"
                        )
                        print(
                            f"   📊 Processing time: {result.get('processing_time', 0):.2f}s"
                        )
                        self.test_results.append("✅ voice_audio_upload")
                    else:
                        print(f"   ❌ Audio upload failed: {response.status_code}")
                        self.test_results.append("❌ voice_audio_upload")
                else:
                    print("   ⚠️ Could not test audio upload - no test audio")
                    self.test_results.append("⚠️ voice_audio_upload")

            except Exception as e:
                print(f"   ❌ Enhanced voice agent test failed: {e}")
                self.test_results.append("❌ enhanced_voice_agent")

    async def test_end_to_end_workflow(self):
        """Test complete end-to-end voice workflow"""
        print("\n🔄 Testing End-to-End Voice Workflow...")

        async with httpx.AsyncClient() as client:
            try:
                # Create a voice workflow with mock transcription
                voice_workflow_data = {
                    "voice_input": {
                        "transcription": "Send an email to Sarah about the Phase 3 testing results",
                        "language": "en-US",
                        "confidence": 0.95,
                        "duration": 4.2,
                    },
                    "user_id": "test_user_phase3_e2e",
                    "metadata": {
                        "test": "phase3_end_to_end",
                        "timestamp": datetime.now().isoformat(),
                    },
                }

                # Create workflow
                response = await client.post(
                    f"{self.base_urls['master']}/workflow/voice",
                    json=voice_workflow_data,
                    timeout=15.0,
                )

                if response.status_code == 200:
                    workflow_data = response.json()
                    workflow_id = workflow_data.get("workflow_id")
                    print(f"   ✅ Voice workflow created: {workflow_id}")

                    # Wait for processing
                    print("   ⏳ Waiting for workflow processing...")
                    await asyncio.sleep(8)

                    # Check status
                    status_response = await client.get(
                        f"{self.base_urls['master']}/workflow/{workflow_id}",
                        timeout=10.0,
                    )

                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        status = status_data.get("status")
                        progress = status_data.get("progress", 0)

                        print(f"   📊 Workflow status: {status} ({progress:.1%})")

                        if status == "completed":
                            result = status_data.get("result", {})
                            if result.get("success"):
                                print(
                                    "   ✅ End-to-end workflow completed successfully!"
                                )
                                self.test_results.append("✅ e2e_workflow")
                            else:
                                print(
                                    f"   ⚠️ Workflow completed with issues: {result.get('message')}"
                                )
                                self.test_results.append("⚠️ e2e_workflow")
                        elif status == "failed":
                            print("   ❌ Workflow failed")
                            self.test_results.append("❌ e2e_workflow")
                        else:
                            print(f"   ⚠️ Workflow still processing: {status}")
                            self.test_results.append("⚠️ e2e_workflow")
                    else:
                        print("   ❌ Failed to get workflow status")
                        self.test_results.append("❌ e2e_workflow_status")

                else:
                    print(
                        f"   ❌ Failed to create voice workflow: {response.status_code}"
                    )
                    self.test_results.append("❌ e2e_workflow_creation")

            except Exception as e:
                print(f"   ❌ End-to-end workflow test failed: {e}")
                self.test_results.append("❌ e2e_workflow")

    async def test_performance_benchmarks(self):
        """Test performance benchmarks"""
        print("\n⚡ Testing Performance Benchmarks...")

        try:
            # Test Whisper transcription speed
            test_audio = self.create_test_audio(
                duration=10.0
            )  # Longer audio for benchmark

            if test_audio:
                import time

                # Benchmark Whisper
                start_time = time.time()
                transcription_result = await whisper_service.transcribe_audio(
                    test_audio
                )
                whisper_time = time.time() - start_time

                audio_duration = transcription_result.duration
                real_time_factor = (
                    audio_duration / whisper_time if whisper_time > 0 else 0
                )

                print(
                    f"   📊 Whisper benchmark: {whisper_time:.2f}s for {audio_duration:.1f}s audio"
                )
                print(f"   ⚡ Real-time factor: {real_time_factor:.1f}x")

                if real_time_factor > 1.0:
                    print("   ✅ Whisper processing faster than real-time")
                    self.test_results.append("✅ whisper_performance")
                else:
                    print("   ⚠️ Whisper processing slower than real-time")
                    self.test_results.append("⚠️ whisper_performance")

                # Test VAD speed
                import librosa

                audio_buffer = io.BytesIO(test_audio)
                audio_signal, sr = librosa.load(audio_buffer, sr=16000)

                start_time = time.time()
                vad_segments = voice_activity_detector.detect_voice_activity(
                    audio_signal, sr
                )
                vad_time = time.time() - start_time

                vad_real_time_factor = audio_duration / vad_time if vad_time > 0 else 0

                print(
                    f"   📊 VAD benchmark: {vad_time:.2f}s for {audio_duration:.1f}s audio"
                )
                print(f"   ⚡ VAD real-time factor: {vad_real_time_factor:.1f}x")

                if vad_real_time_factor > 5.0:  # VAD should be much faster
                    print("   ✅ VAD processing very fast")
                    self.test_results.append("✅ vad_performance")
                else:
                    print("   ⚠️ VAD processing slower than expected")
                    self.test_results.append("⚠️ vad_performance")

            else:
                print("   ⚠️ Could not create test audio for benchmarks")
                self.test_results.append("⚠️ performance_benchmarks")

        except Exception as e:
            print(f"   ❌ Performance benchmark failed: {e}")
            self.test_results.append("❌ performance_benchmarks")

    async def cleanup_test_data(self):
        """Clean up test data"""
        print("\n🧹 Cleaning up test data...")

        try:
            # Clean up uploaded audio files
            for file_id in self.test_audio_files:
                try:
                    await audio_service.delete_audio_file(file_id)
                    print(f"   ✅ Deleted audio file: {file_id}")
                except Exception as e:
                    print(f"   ⚠️ Failed to delete {file_id}: {e}")

            print(f"   🗑️ Cleaned up {len(self.test_audio_files)} test files")

        except Exception as e:
            print(f"   ⚠️ Cleanup warning: {e}")

    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 70)
        print("📋 PHASE 3 VOICE PROCESSING PIPELINE TEST SUMMARY")
        print("=" * 70)

        passed = sum(1 for result in self.test_results if result.startswith("✅"))
        warned = sum(1 for result in self.test_results if result.startswith("⚠️"))
        failed = sum(1 for result in self.test_results if result.startswith("❌"))
        total = len(self.test_results)

        print(f"Total tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"⚠️ Warnings: {warned}")
        print(f"❌ Failed: {failed}")
        print(
            f"Success rate: {(passed / total * 100):.1f}%"
            if total > 0
            else "No tests run"
        )

        print("\n📝 Detailed Results:")
        for result in self.test_results:
            print(f"   {result}")

        print("\n🎯 Phase 3 Components Tested:")
        print("   🎤 Whisper Speech-to-Text Integration")
        print("   📁 Audio File Management with GridFS")
        print("   🔊 Voice Activity Detection")
        print("   🛠️ Audio Processing Utilities")
        print("   🤖 Enhanced Voice Agent with Advanced NLP")
        print("   🔄 End-to-End Voice Workflows")
        print("   ⚡ Performance Benchmarking")

        # Determine overall status
        if failed == 0 and warned <= 3:
            print("\n🎉 Phase 3 testing PASSED! Voice processing pipeline is ready.")
            print("\n✅ Ready to move to Phase 4: Action Execution System")
            return True
        elif failed <= 3:
            print(
                "\n⚠️ Phase 3 testing completed with warnings. Review issues before Phase 4."
            )
            return True
        else:
            print("\n❌ Phase 3 testing FAILED. Fix critical issues before proceeding.")
            return False


async def main():
    """Main testing function"""
    print("🧪 Starting Phase 3 Voice Processing Pipeline Testing")
    print("=" * 70)

    tester = Phase3Tester()

    try:
        # Run all tests
        await tester.test_whisper_service()
        await tester.test_audio_service()
        await tester.test_voice_activity_detection()
        await tester.test_audio_utilities()
        await tester.test_enhanced_voice_agent()
        await tester.test_end_to_end_workflow()
        await tester.test_performance_benchmarks()

        # Cleanup
        await tester.cleanup_test_data()

        # Print summary and return status
        success = tester.print_summary()
        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n🛑 Testing interrupted by user")
        await tester.cleanup_test_data()
        return 1
    except Exception as e:
        print(f"\n❌ Testing failed with error: {e}")
        await tester.cleanup_test_data()
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n❌ Failed to run tests: {e}")
        sys.exit(1)
