import os
import tempfile
from typing import List
import streamlit as st
from PIL import Image
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

# -----------------------------
# Caching model & processor
# -----------------------------
@st.cache_resource(show_spinner=True)
def get_model_and_processor(model_name: str):
    model, processor = load(model_name)
    config = load_config(model_name)
    return model, processor, config

# -----------------------------
# Image utilities
# -----------------------------
DEFAULT_RESIZE = (1024, 1024)

def resize_image_pil(img: Image.Image, size=DEFAULT_RESIZE) -> Image.Image:
    img = img.convert("RGB")
    img.thumbnail(size)
    return img

def save_temp_image(img: Image.Image) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img.save(tmp.name, format="JPEG")
    return tmp.name

# -----------------------------
# Prompt template
# -----------------------------
DEFAULT_PROMPT = """Extract all clearly printed text and structured fields from this trading card image. Focus on the following details:\n\t•\tCard Title (e.g., player name)\n\t•\tPlayer’s Team\n\t•\tCard Series/Release (e.g., 2021 Topps Fire)\n\t•\tCard Subset (e.g., Scorching Signatures)\n\t•\tPSA or grading company label and grade (e.g., PSA 9 MINT)\n\t•\tCard number/serial (e.g., #SSVGJ or other identifier)\n\t•\tCertification or serial number (e.g., 122390435)\n\t•\tAny autograph details (if present, confirm it says ‘Certified Autograph Issue’)\n\t•\tAny visible text overlay or background text\n\nReturn results as a structured JSON object with fields: {\"Player Name\": \"\",\"Team\": \"\",\"Card Series\": \"\",\"Card Subset\": \"\",\"Grading Company\": \"\",\"Grade\": \"\",\"Card Number\": \"\",\"Certification Number\": \"\",\"Autograph Details\": \"\",\"Overlay Text\": \"\"}. If a field is not visible, set its value to ‘Not visible’."""

# -----------------------------
# App Layout
# -----------------------------
st.set_page_config(page_title="Trading Card OCR • MLX", layout="wide")
st.title("Trading Card OCR (MLX Nanonets OCR Model)")

with st.sidebar:
    st.header("Configuration")
    model_name = st.text_input("Model Name", value="mlx-community/Nanonets-OCR2-3B-4bit")
    max_tokens = st.slider("Max Tokens", min_value=32, max_value=512, value=256, step=32)
    run_button_label = st.text_input("Run Button Label", value="Run OCR")
    custom_prompt = st.text_area("Prompt", value=DEFAULT_PROMPT, height=300)

st.markdown("Upload one or more trading card images (front/back). The model will attempt structured extraction.")
uploaded_files = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    cols = st.columns(min(len(uploaded_files), 4))
    temp_paths: List[str] = []
    for i, uf in enumerate(uploaded_files):
        try:
            img = Image.open(uf)
            resized = resize_image_pil(img)
            p = save_temp_image(resized)
            temp_paths.append(p)
            with cols[i % len(cols)]:
                # Updated deprecated parameter use_column_width -> use_container_width
                st.image(resized, caption=uf.name, width='stretch')
        except Exception as e:
            st.error(f"Failed to process {uf.name}: {e}")

    if st.button(run_button_label, type="primary"):
        with st.spinner("Loading model & running inference..."):
            try:
                model, processor, config = get_model_and_processor(model_name)

                formatted_prompt = apply_chat_template(
                    processor, config, custom_prompt, num_images=len(temp_paths)
                )

                result = generate(
                    model,
                    processor,
                    formatted_prompt,
                    temp_paths,
                    max_tokens=max_tokens,
                    verbose=False,
                )

                # NEW: Handle GenerationResult objects
                raw_text = getattr(result, "text", result)
                st.subheader("Raw Model Output")
                st.code(raw_text)

                # Display meta if available
                meta_cols = st.columns(5)
                for col, label, attr in [
                    (meta_cols[0], "Prompt Tokens", "prompt_tokens"),
                    (meta_cols[1], "Generation Tokens", "generation_tokens"),
                    (meta_cols[2], "Total Tokens", "total_tokens"),
                    (meta_cols[3], "Prompt TPS", "prompt_tps"),
                    (meta_cols[4], "Gen TPS", "generation_tps"),
                ]:
                    if hasattr(result, attr):
                        col.metric(label, getattr(result, attr))

                # Attempt to parse JSON substring if present
                import json, re

                def clean_fences(text: str) -> str:
                    # Strip markdown code fences
                    text = re.sub(r"^```(?:json)?\n", "", text.strip())
                    text = re.sub(r"```$", "", text)
                    return text.strip()

                def repair_malformed_json(text: str) -> str:
                    """Repair JSON where stray quoted string lines appear without keys.
                    Returns a JSON string.
                    """
                    kv_pattern = re.compile(r'^"([^"\\]+)"\s*:\s*"([^"\\]*)"$')
                    extra_pattern = re.compile(r'^"([^"\\]+)"$')
                    data = {}
                    extras = []
                    # Remove surrounding braces if present
                    inner = text.strip()
                    if inner.startswith('{') and inner.endswith('}'):
                        inner = inner[1:-1]
                    for raw_line in inner.splitlines():
                        line = raw_line.strip().rstrip(',')
                        if not line:
                            continue
                        m = kv_pattern.match(line)
                        if m:
                            data[m.group(1)] = m.group(2)
                        else:
                            m2 = extra_pattern.match(line)
                            if m2:
                                extras.append(m2.group(1))
                    if extras:
                        data["Unlabeled"] = extras
                    return json.dumps(data, indent=2)

                json_block = None
                parsed = None
                try:
                    # Prefer fenced JSON block
                    fenced = re.search(r"```json\n([\s\S]*?)\n```", raw_text)
                    if fenced:
                        candidate = clean_fences(fenced.group(0))
                        try:
                            parsed = json.loads(candidate)
                        except json.JSONDecodeError:
                            # Attempt repair if the structure looks like an object
                            if candidate.strip().startswith('{'):
                                repaired = repair_malformed_json(candidate)
                                parsed = json.loads(repaired)
                                st.info("JSON was repaired (unlabeled strings captured under 'Unlabeled').")
                    else:
                        # Find first JSON-like block
                        match = re.search(r"\{[\s\S]*?\}", raw_text)
                        if match:
                            candidate = match.group(0)
                            try:
                                parsed = json.loads(candidate)
                            except json.JSONDecodeError:
                                repaired = repair_malformed_json(candidate)
                                parsed = json.loads(repaired)
                                st.info("JSON was repaired (unlabeled strings captured under 'Unlabeled').")
                    if parsed is not None:
                        st.subheader("Parsed JSON")
                        st.json(parsed)
                    else:
                        st.info("No JSON object detected in output.")
                except Exception as je:
                    st.warning(f"Could not parse or repair JSON: {je}")

            finally:
                # Clean up temp files
                for p in temp_paths:
                    try:
                        os.remove(p)
                    except OSError:
                        pass
else:
    st.info("Awaiting image upload.")

st.caption("Powered by MLX Vision-Language Model • Nanonets OCR variant")
