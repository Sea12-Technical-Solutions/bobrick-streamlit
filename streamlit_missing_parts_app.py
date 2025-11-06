import base64
import io
import json
import os
from pathlib import Path

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


def _load_context_image(context_path: Path) -> Image.Image:
    if not context_path.exists():
        raise FileNotFoundError(f"Context image not found at: {context_path}")
    return Image.open(context_path).convert("RGB")


def _call_openai_vision_compare(upload_data_url: str, context_data_url: str, model: str) -> dict:
    client = OpenAI()

    system_prompt = (
        "You are a precise visual QA assistant. Compare two product images. "
        "Return JSON only with a 'missing_parts' array of strings describing items present in the context image but missing in the uploaded image. "
        "Be concise and specific. If nothing is missing, return an empty array."
    )

    user_instructions = (
        "First image is the context (reference). Second is the user upload to check. "
        "Identify any parts/items that are present in the context image but missing in the uploaded image. "
        "Respond ONLY in the following JSON format: {\n  \"missing_parts\": [\"...\"]\n}"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_instructions},
                    {"type": "image_url", "image_url": {"url": context_data_url}},
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
        if not isinstance(data, dict) or "missing_parts" not in data:
            raise ValueError("JSON does not contain 'missing_parts'.")
        if not isinstance(data["missing_parts"], list):
            raise ValueError("'missing_parts' must be a list.")
        return data
    except Exception:
        # Fallback: wrap raw text
        return {"missing_parts": [f"Model response could not be parsed as JSON: {content}"]}


def main() -> None:
    st.set_page_config(page_title="Napkin Tampon Vendor Assembly Verification", page_icon="🔍", layout="centered")
    st.title("🔍 Napkin Tampon Vendor Assembly Verification")
    st.caption(
        "Upload an image. It will be compared to a hardcoded context image using an Orchestrator vision model."
    )

    # Hardcoded context image path: adjust as needed. Defaults to 'best_sample.png' in this folder.
    default_context_path = Path(__file__).resolve().parent / "subassemblygood.JPG"

    with st.sidebar:
        st.subheader("Settings")
        st.write(
            "Context image is hardcoded. You can change the file at:"
        )
        st.code(str(default_context_path), language="bash")
        model = st.selectbox(
            "Vision model",
            options=["gpt-4o", "gpt-4o-mini"],
            index=1,
            help="Requires a valid OpenAI API key in the .env file (OPENAI_API_KEY).",
        )

    uploaded = st.file_uploader("Upload image to check", type=["png", "jpg", "jpeg", "webp"]) 

    if uploaded is not None:
        try:
            uploaded_img = Image.open(uploaded).convert("RGB")
        except Exception as e:
            st.error(f"Failed to read uploaded image: {e}")
            return

        try:
            context_img = _load_context_image(default_context_path)
        except Exception as e:
            st.error(str(e))
            return

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Context (reference)**")
            st.image(context_img, use_container_width=True)
        with col2:
            st.markdown("**Uploaded**")
            st.image(uploaded_img, use_container_width=True)

        analyze = st.button("Analyze for Missing Parts", type="primary")

        if analyze:
            if not os.getenv("OPENAI_API_KEY"):
                st.error(
                    "OPENAI_API_KEY not set. Please add it to your .env file in the project root directory."
                )
                return

            with st.spinner("Analyzing with Orchestrator Vision..."):
                upload_data_url = _image_to_data_url(uploaded_img)
                context_data_url = _image_to_data_url(context_img)
                try:
                    result = _call_openai_vision_compare(
                        upload_data_url=upload_data_url,
                        context_data_url=context_data_url,
                        model=model,
                    )
                except Exception as e:
                    st.error(f"OpenAI request failed: {e}")
                    return

            missing_parts = result.get("missing_parts", [])
            st.subheader("Differences (Missing Parts)")
            if not missing_parts:
                st.success("No missing parts detected compared to the context image.")
            else:
                for idx, item in enumerate(missing_parts, start=1):
                    st.markdown(f"- {idx}. {item}")


if __name__ == "__main__":
    main()


