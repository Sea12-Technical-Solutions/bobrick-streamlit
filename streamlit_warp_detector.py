#!/usr/bin/env python3
"""
Streamlit Dashboard for Warp Detection

Analyzes a video feed to determine if a part is warped (not flush with conveyor belt)
or not warped (flush with conveyor belt).
Uses temporal summary chain approach: samples frames, describes them with vision LLM,
then synthesizes with text LLM.
"""

import base64
import io
import os
import tempfile
from pathlib import Path

import cv2
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Warp Detection - Top/Bottom Defect Detection",
    page_icon="📏",
    layout="wide"
)


def encode_image(image_array):
    """
    Encode a numpy image array to base64 string.
    
    Args:
        image_array: numpy array (BGR format from cv2)
        
    Returns:
        Base64 encoded string of the image
    """
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    
    # Encode to JPEG
    _, buffer = cv2.imencode('.jpg', image_rgb)
    return base64.b64encode(buffer).decode('utf-8')


def sample_frames(video_path, interval_seconds=1.0):
    """
    Sample frames from video at regular intervals.
    
    Args:
        video_path: Path to video file
        interval_seconds: Interval between sampled frames (default: 1.0 second)
        
    Returns:
        List of tuples (frame_number, timestamp_seconds, frame_array)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(fps * interval_seconds)
    
    sampled_frames = []
    frame_number = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Sample every frame_interval frames
        if frame_number % frame_interval == 0:
            timestamp = frame_number / fps
            sampled_frames.append((frame_number, timestamp, frame))
        
        frame_number += 1
    
    cap.release()
    return sampled_frames


def describe_frame(client, frame_array, timestamp):
    """
    Get a description of a single frame using vision LLM, focusing on flush/warp detection.
    
    Args:
        client: OpenAI client
        frame_array: numpy array of the frame
        timestamp: timestamp in seconds
        
    Returns:
        String description of the frame
    """
    base64_image = encode_image(frame_array)
    
    prompt = f"""Describe what you see in this frame at {timestamp:.1f} seconds. 
Focus on:
1. The part's contact with the conveyor belt - is it flush/flat against the belt surface?
2. Any visible gaps between the part and the conveyor belt
3. Any parts of the part that are lifted or not touching the belt
4. The overall flatness of the part relative to the belt surface

Be concise but specific about whether the part appears flush with the belt or if there are gaps/lifting."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=200,
            temperature=0.3
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error describing frame: {str(e)}"


def analyze_warp(client, frame_descriptions):
    """
    Analyze frame descriptions to determine if the part is warped or flush.
    
    Args:
        client: OpenAI client
        frame_descriptions: List of (timestamp, description) tuples
        
    Returns:
        String analysis result
    """
    all_descriptions_text = "\n\n".join([
        f"At {timestamp:.1f} seconds: {description}"
        for timestamp, description in frame_descriptions
    ])
    
    prompt = f"""You are analyzing a video sequence of a part on a conveyor belt to determine if the part is warped.
Below are descriptions of frames sampled throughout the video:

{all_descriptions_text}

A part is considered WARPED if at any point in the video feed:
- The part is not flush with the conveyor belt
- There are visible gaps between the part and the belt
- Parts of the part are lifted or not touching the belt surface
- The part does not lie flat against the belt

A part is considered NOT WARPED if:
- The part is consistently flush with the conveyor belt throughout the video
- The part lies flat against the belt surface
- There are no visible gaps between the part and the belt
- The part maintains good contact with the belt surface

Based on these descriptions, determine:
1. Is the part WARPED (not flush with belt) at any point in the video?
2. Or is the part NOT WARPED (flush with belt) throughout the video?

Provide your answer in this format:
RESULT: [WARPED or NOT_WARPED]
REASONING: [brief explanation of why, including which frames show any issues if warped]"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=300,
            temperature=0.3
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error analyzing: {str(e)}"


def main():
    """Main Streamlit app function."""
    st.title("Warp Detection - Top/Bottom Defect Detection")
    st.caption(
        "Analyzes a video feed to determine if the panel is warped (not flush with conveyor belt) "
        "or not warped (flush with conveyor belt). Uses Orchestrator Vision to analyze video feed."
    )
    
    # Sidebar settings
    with st.sidebar:
        st.subheader("⚙️ Settings")
        
        model = st.selectbox(
            "Vision Model",
            options=["gpt-4o", "gpt-4o-mini"],
            index=0,
            help="OpenAI model to use for vision analysis. Requires OPENAI_API_KEY in .env file."
        )
        
        interval = st.slider(
            "Frame Sampling Interval (seconds)",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.5,
            help="Interval between sampled frames. Smaller values = more frames analyzed."
        )
        
        st.markdown("---")
        st.markdown("**About**")
        st.info(
            "This tool samples frames from your video, describes them using a vision model, "
            "then analyzes the descriptions to determine if the part is warped (not flush) "
            "or not warped (flush with the conveyor belt)."
        )
    
    # Video upload
    uploaded_video = st.file_uploader(
        "Upload a video file of a part on a conveyor belt",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        help="Supported formats: MP4, AVI, MOV, MKV, WEBM"
    )
    
    if uploaded_video is not None:
        # Display video info
        st.markdown("### 📹 Uploaded Video")
        
        # Save uploaded video to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_video.name).suffix) as tmp_file:
            tmp_file.write(uploaded_video.read())
            tmp_video_path = tmp_file.name
        
        try:
            # Display video
            video_bytes = uploaded_video.read()
            st.video(video_bytes)
            
            # Analyze button
            analyze_button = st.button("🔍 Analyze Video for Warp", type="primary", use_container_width=True)
            
            if analyze_button:
                if not os.getenv("OPENAI_API_KEY"):
                    st.error(
                        "❌ OPENAI_API_KEY not set. Please add it to your .env file in the project root directory."
                    )
                    return
                
                # Initialize OpenAI client
                client = OpenAI()
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Step 1: Sample frames
                    status_text.text("📊 Sampling frames from video...")
                    sampled_frames = sample_frames(tmp_video_path, interval)
                    
                    if len(sampled_frames) == 0:
                        st.error("❌ No frames could be sampled from video")
                        return
                    
                    st.info(f"✅ Sampled {len(sampled_frames)} frames from the video")
                    
                    # Step 2: Describe frames
                    status_text.text("👁️ Describing frames with vision model...")
                    frame_descriptions = []
                    progress_bar.progress(0)
                    
                    for i, (frame_num, timestamp, frame) in enumerate(sampled_frames, 1):
                        progress = (i - 1) / len(sampled_frames)
                        progress_bar.progress(progress)
                        status_text.text(f"👁️ Processing frame {i}/{len(sampled_frames)} (t={timestamp:.1f}s)...")
                        
                        description = describe_frame(client, frame, timestamp)
                        frame_descriptions.append((timestamp, description))
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Frame descriptions complete!")
                    
                    # Display frame descriptions
                    st.markdown("---")
                    st.subheader("📝 Frame Descriptions")
                    
                    # Show sample frames with descriptions
                    cols = st.columns(min(3, len(sampled_frames)))
                    for idx, ((frame_num, timestamp, frame), (desc_timestamp, description)) in enumerate(
                        zip(sampled_frames, frame_descriptions)
                    ):
                        col_idx = idx % 3
                        with cols[col_idx]:
                            # Convert BGR to RGB for display
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            st.image(frame_rgb, caption=f"t={timestamp:.1f}s", use_container_width=True)
                            st.caption(f"**{timestamp:.1f}s**: {description[:100]}...")
                    
                    # Show all descriptions in expander
                    with st.expander("📋 View All Frame Descriptions"):
                        for timestamp, description in frame_descriptions:
                            st.markdown(f"**At {timestamp:.1f} seconds:**")
                            st.write(description)
                            st.markdown("---")
                    
                    # Step 3: Analyze warp
                    status_text.text("🧠 Analyzing for warp...")
                    progress_bar.progress(0.5)
                    
                    result = analyze_warp(client, frame_descriptions)
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Analysis complete!")
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("🎯 Analysis Result")
                    
                    # Parse result
                    result_lines = result.split('\n')
                    result_type = None
                    reasoning = []
                    
                    for line in result_lines:
                        if line.startswith('RESULT:'):
                            result_type = line.replace('RESULT:', '').strip()
                        elif line.startswith('REASONING:'):
                            reasoning.append(line.replace('REASONING:', '').strip())
                        elif reasoning:
                            reasoning.append(line.strip())
                    
                    # Display result with appropriate styling
                    if result_type:
                        if 'WARPED' in result_type.upper():
                            st.error(f"⚠️ **Result: {result_type}**")
                            st.warning("The part is not flush with the conveyor belt and is considered warped.")
                        elif 'NOT_WARPED' in result_type.upper() or 'NOT WARPED' in result_type.upper():
                            st.success(f"✅ **Result: {result_type}**")
                            st.success("The part is flush with the conveyor belt and is not warped.")
                        else:
                            st.write(f"**Result: {result_type}**")
                    
                    if reasoning:
                        st.markdown("**Reasoning:**")
                        st.write('\n'.join(reasoning))
                    
                    # Show full raw result in expander
                    with st.expander("📄 View Full Analysis"):
                        st.code(result)
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    st.exception(e)
                
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(tmp_video_path)
                    except:
                        pass
        
        except Exception as e:
            st.error(f"❌ Error processing video: {str(e)}")
            st.exception(e)
            
            # Clean up temporary file
            try:
                os.unlink(tmp_video_path)
            except:
                pass


if __name__ == "__main__":
    main()

