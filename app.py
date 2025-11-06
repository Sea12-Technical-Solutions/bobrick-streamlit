#!/usr/bin/env python3
"""
Main Streamlit App - Multi-Page Application
Combines all 5 Streamlit apps into a single deployment
"""

import streamlit as st

# Page config (must be first Streamlit command)
st.set_page_config(
    page_title="Quality Control Apps",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Navigation
st.sidebar.title("🔍 Quality Control Apps")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select an app:",
    [
        "🏠 Home",
        "📏 Warp Detector",
        "🔍 Chrome Scratch Detector",
        "📹 Video Rotation Analyzer",
        "🔧 Missing Parts App",
        "🔧 Missing Parts App 2"
    ]
)

if page == "🏠 Home":
    st.title("Welcome to Quality Control Apps")
    st.markdown("""
    This application contains 5 quality control tools:
    
    1. **📏 Warp Detector** - Analyzes video feeds to detect if parts are warped (not flush with conveyor belt)
    2. **🔍 Chrome Scratch Detector** - Detects scratches on chrome plating using vision AI
    3. **📹 Video Rotation Analyzer** - Analyzes video rotation and orientation
    4. **🔧 Missing Parts App** - Detects missing parts in assemblies
    5. **🔧 Missing Parts App 2** - Alternative missing parts detection
    
    Select an app from the sidebar to get started.
    """)
    
    # Check for API key
    from utils import get_openai_api_key
    api_key = get_openai_api_key()
    if not api_key:
        st.warning("⚠️ **OPENAI_API_KEY not found.** Please set it in your environment variables or Streamlit Cloud secrets.")
    else:
        st.success("✅ OpenAI API key is configured.")
    
elif page == "📏 Warp Detector":
    # Import and run warp detector (lazy import)
    try:
        import streamlit_warp_detector
        streamlit_warp_detector.main()
    except ImportError as e:
        st.error(f"Failed to import warp detector: {e}")
        st.info("Make sure all dependencies are installed, including opencv-python-headless")
    
elif page == "🔍 Chrome Scratch Detector":
    # Import and run chrome scratch detector
    try:
        import streamlit_chrome_scratch_detector
        streamlit_chrome_scratch_detector.main()
    except ImportError as e:
        st.error(f"Failed to import chrome scratch detector: {e}")
    
elif page == "📹 Video Rotation Analyzer":
    # Import and run video rotation analyzer (lazy import)
    try:
        import streamlit_video_rotation_analyzer
        streamlit_video_rotation_analyzer.main()
    except ImportError as e:
        st.error(f"Failed to import video rotation analyzer: {e}")
        st.info("Make sure all dependencies are installed, including opencv-python-headless")
    
elif page == "🔧 Missing Parts App":
    # Import and run missing parts app
    try:
        import streamlit_missing_parts_app
        streamlit_missing_parts_app.main()
    except ImportError as e:
        st.error(f"Failed to import missing parts app: {e}")
    
elif page == "🔧 Missing Parts App 2":
    # Import and run missing parts app 2
    try:
        import streamlit_missing_parts_app2
        streamlit_missing_parts_app2.main()
    except ImportError as e:
        st.error(f"Failed to import missing parts app 2: {e}")

