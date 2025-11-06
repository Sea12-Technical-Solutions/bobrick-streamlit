#!/usr/bin/env python3
"""
Streamlit Dashboard for Video Rotation Analyzer

Analyzes a video to determine if a worker rotates a part or just slides it along.
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
    page_title="Sequence Verification - Rotation Detection",
    page_icon="🎬",
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
    Get a description of a single frame using vision LLM.
    
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
1. The position and orientation of the part/object
2. The worker's hands/actions
3. Any movement or change happening

Be concise but specific about the part's orientation and position."""

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


def analyze_rotation(client, frame_descriptions, min_time_seconds=2.0):
    """
    Analyze frame descriptions to determine if rotation or sliding occurred.
    
    Args:
        client: OpenAI client
        frame_descriptions: List of (timestamp, description) tuples
        min_time_seconds: Minimum timestamp (in seconds) to consider for rotation detection
        
    Returns:
        String analysis result
    """
    # Filter frames: all frames for context, but only frames after min_time_seconds for rotation analysis
    all_descriptions_text = "\n\n".join([
        f"At {timestamp:.1f} seconds: {description}"
        for timestamp, description in frame_descriptions
    ])
    
    # Separate frames before and after the minimum time
    frames_before_min = [(t, d) for t, d in frame_descriptions if t < min_time_seconds]
    frames_after_min = [(t, d) for t, d in frame_descriptions if t >= min_time_seconds]
    
    if len(frames_after_min) == 0:
        return "RESULT: INSUFFICIENT_DATA\nREASONING: No frames found after the minimum time threshold for analysis."
    
    # Build descriptions for frames after minimum time
    relevant_descriptions_text = "\n\n".join([
        f"At {timestamp:.1f} seconds: {description}"
        for timestamp, description in frames_after_min
    ])
    
    prompt = f"""You are analyzing a video sequence to determine how a worker moves a part. 
Below are descriptions of frames sampled throughout the video:

{all_descriptions_text}

IMPORTANT: Only consider rotations that occur at or after {min_time_seconds} seconds. 
Any rotations or orientation changes that happen before {min_time_seconds} seconds should be IGNORED.

For your analysis, focus on frames from {min_time_seconds} seconds onwards:
{relevant_descriptions_text}

Based on these descriptions (only considering frames at or after {min_time_seconds} seconds), determine:
1. Does the worker ROTATE the part (change its orientation/rotation) at or after {min_time_seconds} seconds?
2. Or does the worker just SLIDE the part along (move it without changing orientation)?

Look for changes in orientation between frames at or after {min_time_seconds} seconds. 
If the part's orientation changes significantly after this time threshold, it's rotation. 
If the part moves but maintains the same orientation, it's sliding.

Provide your answer in this format:
RESULT: [ROTATION or SLIDING]
REASONING: [brief explanation of why]"""

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
    st.title("Sequence Verification - Rotation Detection")
    st.caption(
        "Analyzes a video to determine if a worker rotates a part or just slides it along. "
        "Uses Orchestrator Vision to analyze video feed."
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
            value=1.0,
            step=0.5,
            help="Interval between sampled frames. Smaller values = more frames analyzed."
        )
        
        min_time = st.slider(
            "Minimum Time Threshold (seconds)",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.5,
            help="Rotations before this time are ignored. Only consider rotations after this threshold."
        )
        
        st.markdown("---")
        st.markdown("**About**")
        st.info(
            "This tool samples frames from your video, describes them using a vision model, "
            "then analyzes the descriptions to determine if rotation or sliding occurred."
        )
    
    # Video upload
    uploaded_video = st.file_uploader(
        "Upload a video file",
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
            analyze_button = st.button("🔍 Analyze Video", type="primary", use_container_width=True)
            
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
                    
                    # Step 3: Analyze rotation
                    status_text.text("🧠 Analyzing motion pattern...")
                    progress_bar.progress(0.5)
                    
                    result = analyze_rotation(client, frame_descriptions, min_time_seconds=min_time)
                    
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
                        if 'ROTATION' in result_type.upper():
                            st.success(f"🔄 **Result: {result_type}**")
                        elif 'SLIDING' in result_type.upper():
                            st.info(f"➡️ **Result: {result_type}**")
                        elif 'INSUFFICIENT' in result_type.upper():
                            st.warning(f"⚠️ **Result: {result_type}**")
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

