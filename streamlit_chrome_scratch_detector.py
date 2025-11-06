import base64
import io
import json
import os

import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()


def _image_to_data_url(img: Image.Image, format_hint: str | None = None) -> str:
    """
    Convert a PIL Image to a data URL suitable for OpenAI vision inputs.
    """
    buffer = io.BytesIO()
    format_to_use = format_hint or (img.format if img.format else "PNG")
    img.save(buffer, format=format_to_use)
    mime = "image/png" if format_to_use.upper() == "PNG" else f"image/{format_to_use.lower()}"
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _call_openai_vision_scratch_detection(upload_data_url: str, model: str) -> dict:
    """
    Send image to OpenAI Vision API and ask it to check for scratches on chrome plating.
    """
    client = OpenAI()

    system_prompt = (
        "You are a quality control inspector specializing in chrome plating surfaces. "
        "Your task is to carefully examine images for scratches, defects, or damage on chrome-plated surfaces. "
        "Return JSON only with the following structure: "
        "{\n  \"scratches_found\": boolean,\n  \"scratch_count\": number,\n  \"scratch_details\": [\n    {\n      \"location\": string,\n      \"description\": string,\n      \"severity\": string\n    }\n  ],\n  \"overall_assessment\": string\n}. "
        "Be precise and thorough in your analysis. If no scratches are found, set scratches_found to false and return an empty array for scratch_details."
    )

    user_instructions = (
        "Please examine this image carefully for any scratches, marks, or defects on chrome plating surfaces. "
        "Chrome plating typically has a shiny, reflective, mirror-like finish. Look for: "
        "- Scratches (long thin lines or marks) "
        "- Scuffs (surface abrasions) "
        "- Dents or dings "
        "- Discoloration or tarnishing "
        "- Any other surface defects that would compromise the chrome finish\n\n"
        "For each scratch or defect found, provide: "
        "- Location: where on the surface it appears (e.g., 'top-left corner', 'center area', 'bottom edge') "
        "- Description: detailed description of the defect "
        "- Severity: 'Cat 3', 'Cat 2', or 'Cat 1' where Cat 1 is the most severe and Cat 3 is the least severe\n\n"
        "Provide an overall assessment of the chrome plating condition. "
        "Respond ONLY in the JSON format specified above. Do not include any other text."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_instructions},
                    {"type": "image_url", "image_url": {"url": upload_data_url}},
                ],
            },
        ],
        temperature=0.0,
    )

    content = response.choices[0].message.content.strip()

    # Handle potential code fences
    if content.startswith("```"):
        # remove triple backticks and optional language hints
        content = content.strip("`\n ")
        if content.lower().startswith("json\n"):
            content = content[5:]

    try:
        data = json.loads(content)
        # Validate structure
        if not isinstance(data, dict):
            raise ValueError("Response is not a JSON object.")
        if "scratches_found" not in data:
            raise ValueError("JSON does not contain 'scratches_found'.")
        return data
    except Exception as e:
        # Fallback: return error structure
        return {
            "scratches_found": None,
            "scratch_count": 0,
            "scratch_details": [],
            "overall_assessment": f"Error parsing model response: {str(e)}. Raw response: {content[:200]}",
        }


def main() -> None:
    st.set_page_config(
        page_title="Chrome Plating Scratch Detector", page_icon="🔍", layout="centered"
    )
    st.title("🔍 Chrome Plating Scratch Detector")
    st.caption(
        "Upload an image of chrome-plated surfaces to detect scratches and defects using a vision model."
    )

    with st.sidebar:
        st.subheader("Settings")
        model = st.selectbox(
            "Vision model",
            options=["gpt-4o", "gpt-4o-mini"],
            index=1,
            help="Requires a valid OpenAI API key in the .env file (OPENAI_API_KEY).",
        )
        st.markdown("---")
        st.markdown("**About**")
        st.info(
            "This tool analyzes images for scratches, scuffs, and other defects on chrome-plated surfaces. "
            "Upload a clear image of the chrome surface for best results."
        )

    uploaded = st.file_uploader(
        "Upload image to analyze", type=["png", "jpg", "jpeg", "webp"]
    )

    if uploaded is not None:
        try:
            uploaded_img = Image.open(uploaded).convert("RGB")
        except Exception as e:
            st.error(f"Failed to read uploaded image: {e}")
            return

        st.markdown("### Uploaded Image")
        st.image(uploaded_img, use_container_width=True)

        analyze = st.button("🔍 Analyze for Scratches", type="primary", use_container_width=True)

        if analyze:
            if not os.getenv("OPENAI_API_KEY"):
                st.error(
                    "OPENAI_API_KEY not set. Please add it to your .env file in the project root directory."
                )
                return

            with st.spinner("Analyzing image for scratches and defects..."):
                upload_data_url = _image_to_data_url(uploaded_img)
                try:
                    result = _call_openai_vision_scratch_detection(
                        upload_data_url=upload_data_url,
                        model=model,
                    )
                except Exception as e:
                    st.error(f"Vision API request failed: {e}")
                    return

            # Display results
            st.markdown("---")
            st.subheader("📊 Analysis Results")

            scratches_found = result.get("scratches_found")
            scratch_count = result.get("scratch_count", 0)
            scratch_details = result.get("scratch_details", [])
            overall_assessment = result.get("overall_assessment", "No assessment provided.")

            if scratches_found is None:
                # Error case
                st.error("❌ Analysis Error")
                st.text(overall_assessment)
            elif scratches_found is False or scratch_count == 0:
                st.success("✅ No scratches detected!")
                st.info("The chrome plating appears to be in good condition with no visible scratches or defects.")
            else:
                st.warning(f"⚠️ Scratches detected: {scratch_count} found")
                
                # Display overall assessment
                st.markdown("**Overall Assessment:**")
                st.info(overall_assessment)

                # Display detailed scratch information
                if scratch_details:
                    st.markdown("**Detailed Scratch Information:**")
                    for idx, scratch in enumerate(scratch_details, start=1):
                        with st.expander(f"Scratch {idx}: {scratch.get('location', 'Unknown location')}"):
                            severity = scratch.get("severity", "unknown").upper()
                            severity_color = {
                                "MINOR": "🟢",
                                "MODERATE": "🟡",
                                "SEVERE": "🔴"
                            }.get(severity, "⚪")
                            
                            st.markdown(f"**Severity:** {severity_color} {severity}")
                            st.markdown(f"**Location:** {scratch.get('location', 'Not specified')}")
                            st.markdown(f"**Description:** {scratch.get('description', 'No description provided')}")


if __name__ == "__main__":
    main()

