"""
Streamlit Voice Interface for Phase 3
Real-time voice recording and processing interface
"""

import streamlit as st
import requests
import json
import time
import io
from datetime import datetime
import pandas as pd
from typing import Dict, Any, Optional

# Audio recording components
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    st.warning(
        "streamlit-webrtc not available. Install with: pip install streamlit-webrtc"
    )

# Configuration
AGENT_URLS = {
    "master": "http://localhost:8000",
    "voice": "http://localhost:8001",
    "action": "http://localhost:8002",
}


def initialize_session_state():
    """Initialize Streamlit session state"""
    if "workflows" not in st.session_state:
        st.session_state.workflows = []
    if "current_recording" not in st.session_state:
        st.session_state.current_recording = None
    if "processing_status" not in st.session_state:
        st.session_state.processing_status = None
    if "user_id" not in st.session_state:
        st.session_state.user_id = f"streamlit_user_{int(time.time())}"


def check_agent_health():
    """Check if all agents are healthy"""
    health_status = {}

    for agent_name, url in AGENT_URLS.items():
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                health_status[agent_name] = {
                    "status": health_data.get("status", "unknown"),
                    "uptime": health_data.get("uptime", 0),
                }
            else:
                health_status[agent_name] = {"status": "unhealthy", "uptime": 0}
        except Exception as e:
            health_status[agent_name] = {
                "status": "error",
                "uptime": 0,
                "error": str(e),
            }

    return health_status


def display_agent_status():
    """Display agent health status"""
    st.sidebar.subheader("🏥 Agent Health")

    health_status = check_agent_health()

    for agent_name, status in health_status.items():
        agent_status = status.get("status", "unknown")
        uptime = status.get("uptime", 0)

        if agent_status == "healthy":
            st.sidebar.success(
                f"✅ {agent_name.title()}: {agent_status} ({uptime:.0f}s)"
            )
        elif agent_status == "unhealthy":
            st.sidebar.warning(f"⚠️ {agent_name.title()}: {agent_status}")
        else:
            st.sidebar.error(f"❌ {agent_name.title()}: {agent_status}")


def create_text_workflow(text: str, user_id: str) -> Dict[str, Any]:
    """Create a text-based workflow"""
    try:
        response = requests.post(
            f"{AGENT_URLS['master']}/workflow/text",
            json={
                "text": text,
                "user_id": user_id,
                "metadata": {
                    "source": "streamlit_interface",
                    "timestamp": datetime.now().isoformat(),
                },
            },
            timeout=30,
        )

        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}: {response.text}"}

    except Exception as e:
        return {"error": str(e)}


def upload_audio_workflow(
    audio_bytes: bytes, filename: str, user_id: str
) -> Dict[str, Any]:
    """Upload audio and create workflow"""
    try:
        # Upload to voice agent
        files = {"audio_file": (filename, io.BytesIO(audio_bytes), "audio/wav")}
        data = {"user_id": user_id, "enable_vad": True, "include_segments": True}

        response = requests.post(
            f"{AGENT_URLS['voice']}/upload_audio", files=files, data=data, timeout=60
        )

        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}: {response.text}"}

    except Exception as e:
        return {"error": str(e)}


def get_workflow_status(workflow_id: str) -> Dict[str, Any]:
    """Get workflow status"""
    try:
        response = requests.get(
            f"{AGENT_URLS['master']}/workflow/{workflow_id}", timeout=10
        )

        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}: {response.text}"}

    except Exception as e:
        return {"error": str(e)}


def display_workflow_result(workflow_data: Dict[str, Any]):
    """Display workflow results"""
    if "error" in workflow_data:
        st.error(f"❌ Error: {workflow_data['error']}")
        return

    # Basic info
    workflow_id = workflow_data.get("workflow_id", "unknown")
    status = workflow_data.get("status", "unknown")
    progress = workflow_data.get("progress", 0.0)

    # Status display
    if status == "completed":
        st.success(f"✅ Workflow completed!")
    elif status == "processing":
        st.info(f"🔄 Processing... ({progress:.1%})")
    elif status == "failed":
        st.error(f"❌ Workflow failed")
    else:
        st.warning(f"⚠️ Status: {status}")

    # Progress bar
    if status == "processing":
        st.progress(progress)

    # Results
    result = workflow_data.get("result")
    if result:
        st.subheader("📋 Results")

        if result.get("success"):
            st.success(f"✅ {result.get('message', 'Success')}")

            # Display result data
            result_data = result.get("data", {})
            if result_data:
                st.json(result_data)
        else:
            st.error(f"❌ {result.get('message', 'Failed')}")


def voice_recording_interface():
    """Voice recording interface"""
    st.subheader("🎤 Voice Recording")

    if not WEBRTC_AVAILABLE:
        st.error("Voice recording requires streamlit-webrtc. Please install it.")
        return

    # Recording configuration
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    # WebRTC streamer for audio recording
    webrtc_ctx = webrtc_streamer(
        key="voice-recording",
        mode=WebRtcMode.SENDONLY,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": False, "audio": True},
        audio_receiver_size=1024,
    )

    if webrtc_ctx.audio_receiver:
        st.info("🎙️ Recording audio... Click 'STOP' when finished.")

        # Collect audio frames
        audio_frames = []
        while True:
            try:
                audio_frame = webrtc_ctx.audio_receiver.get_frame(timeout=1)
                if audio_frame:
                    audio_frames.append(audio_frame)
            except:
                break

        if audio_frames:
            st.success(f"📊 Recorded {len(audio_frames)} audio frames")

            # Process audio frames (simplified)
            # In a real implementation, you'd convert frames to audio bytes
            st.info("🔄 Audio processing would happen here...")

            # For demo purposes, we'll simulate audio processing
            if st.button("Process Recording"):
                st.session_state.processing_status = "processing"
                st.rerun()


def text_input_interface():
    """Text input interface"""
    st.subheader("💬 Text Input")

    # Text input
    text_input = st.text_area(
        "Enter your command:",
        placeholder="Example: Schedule a meeting with John tomorrow at 2 PM",
        height=100,
    )

    # Process button
    col1, col2 = st.columns([1, 4])

    with col1:
        if st.button("🚀 Process", type="primary", disabled=not text_input.strip()):
            if text_input.strip():
                # Create workflow
                with st.spinner("Creating workflow..."):
                    result = create_text_workflow(text_input, st.session_state.user_id)

                # Store result
                workflow_data = {
                    "id": result.get("workflow_id", f"wf_{int(time.time())}"),
                    "type": "text",
                    "input": text_input,
                    "timestamp": datetime.now(),
                    "result": result,
                }

                st.session_state.workflows.append(workflow_data)
                st.session_state.processing_status = "completed"
                st.rerun()

    with col2:
        if st.button("🗑️ Clear"):
            st.rerun()


def audio_upload_interface():
    """Audio file upload interface"""
    st.subheader("📁 Audio Upload")

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload audio file",
        type=["wav", "mp3", "m4a", "ogg", "flac"],
        help="Supported formats: WAV, MP3, M4A, OGG, FLAC",
    )

    if uploaded_file:
        # Display file info
        st.info(f"📄 File: {uploaded_file.name} ({uploaded_file.size:,} bytes)")

        # Audio player
        st.audio(uploaded_file.getvalue())

        # Process button
        if st.button("🎵 Process Audio", type="primary"):
            with st.spinner("Processing audio..."):
                # Read file
                audio_bytes = uploaded_file.getvalue()

                # Upload and process
                result = upload_audio_workflow(
                    audio_bytes, uploaded_file.name, st.session_state.user_id
                )

                # Store result
                workflow_data = {
                    "id": result.get("workflow_id", f"wf_{int(time.time())}"),
                    "type": "audio",
                    "input": uploaded_file.name,
                    "timestamp": datetime.now(),
                    "result": result,
                }

                st.session_state.workflows.append(workflow_data)
                st.session_state.processing_status = "completed"
                st.rerun()


def workflow_history():
    """Display workflow history"""
    st.subheader("📊 Workflow History")

    if not st.session_state.workflows:
        st.info("No workflows yet. Try processing some voice commands or text!")
        return

    # Display workflows
    for i, workflow in enumerate(reversed(st.session_state.workflows)):
        with st.expander(
            f"Workflow {workflow['id'][:8]}... ({workflow['type']}) - {workflow['timestamp'].strftime('%H:%M:%S')}"
        ):
            # Input info
            st.write(f"**Input:** {workflow['input']}")
            st.write(f"**Type:** {workflow['type'].title()}")
            st.write(f"**Timestamp:** {workflow['timestamp']}")

            # Result
            result = workflow.get("result", {})

            if "error" in result:
                st.error(f"❌ Error: {result['error']}")
            else:
                # Check current status if workflow ID available
                workflow_id = result.get("workflow_id")
                if workflow_id:
                    with st.spinner("Checking status..."):
                        current_status = get_workflow_status(workflow_id)
                        display_workflow_result(current_status)
                else:
                    # Display direct result
                    if result:
                        st.json(result)


def system_metrics():
    """Display system metrics"""
    st.subheader("📈 System Metrics")

    try:
        # Get metrics from master agent
        response = requests.get(f"{AGENT_URLS['master']}/metrics", timeout=10)

        if response.status_code == 200:
            metrics = response.json()

            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Workflows", metrics.get("total_workflows", 0))

            with col2:
                st.metric("Recent (24h)", metrics.get("recent_workflows_24h", 0))

            with col3:
                success_rate = metrics.get("success_rate_24h", 0)
                st.metric("Success Rate", f"{success_rate:.1%}")

            with col4:
                st.metric("Active", metrics.get("active_workflows", 0))

            # Additional metrics
            if st.checkbox("Show detailed metrics"):
                st.json(metrics)
        else:
            st.error(f"Failed to get metrics: HTTP {response.status_code}")

    except Exception as e:
        st.error(f"Metrics error: {e}")


def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="Voice-to-Action Interface",
        page_icon="🎤",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize session state
    initialize_session_state()

    # Title
    st.title("🎤 Voice-to-Action Interface")
    st.markdown("**Real Voice Processing Pipeline**")

    # Sidebar
    with st.sidebar:
        st.header("🎛️ Control Panel")

        # User ID
        st.text_input("User ID", value=st.session_state.user_id, disabled=True)

        # Agent status
        display_agent_status()

        # Settings
        st.subheader("⚙️ Settings")

        # Language selection
        language = st.selectbox(
            "Language",
            ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
            index=0,
        )

        # Auto-refresh
        auto_refresh = st.checkbox("Auto-refresh", value=False)

        if auto_refresh:
            time.sleep(5)
            st.rerun()

    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "💬 Text Input",
            "📁 Audio Upload",
            "🎙️ Voice Recording",
            "📊 History",
            "📈 Metrics",
        ]
    )

    with tab1:
        text_input_interface()

    with tab2:
        audio_upload_interface()

    with tab3:
        voice_recording_interface()

    with tab4:
        workflow_history()

    with tab5:
        system_metrics()

    # Status updates
    if st.session_state.processing_status:
        if st.session_state.processing_status == "processing":
            st.info("🔄 Processing your request...")
        elif st.session_state.processing_status == "completed":
            st.success("✅ Processing completed!")
            st.session_state.processing_status = None


if __name__ == "__main__":
    main()
