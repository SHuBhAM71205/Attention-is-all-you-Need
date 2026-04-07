from __future__ import annotations

import os
import streamlit as st

from db.Mongo import save_translation_record
from inference import device, load_model, translate_text


st.set_page_config(
    page_title="English to Hindi Translator",
    page_icon="🌐",
    layout="centered",
)

mongo_configured = bool(os.getenv("MONGO_DB"))
if not mongo_configured:
    st.warning(
        "MongoDB is not configured. Translation submissions will not be saved until MONGO_DB is set."
    )

if "translated_options" not in st.session_state:
    st.session_state["translated_options"] = []
if "translation_time_ms" not in st.session_state:
    st.session_state["translation_time_ms"] = None
if "submitted_translation" not in st.session_state:
    st.session_state["submitted_translation"] = None
if "last_source_text" not in st.session_state:
    st.session_state["last_source_text"] = ""

@st.cache_resource(show_spinner=False)
def get_model_bundle():
    return load_model()


st.title("English to Hindi Translator")
st.caption("Transformer demo built from your trained checkpoint.")

with st.sidebar:
    st.subheader("Model")
    st.write(f"Device: `{device}`")

try:
    model, tokenizer, checkpoint_path = get_model_bundle()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()
except Exception as error:  # pragma: no cover - UI fallback
    st.error(f"Unable to load the model: {error}")
    st.stop()

with st.sidebar:
    st.write(f"Checkpoint: `{checkpoint_path}`")

sample_text = "This ceremony took place three weeks ago."

source_text = st.text_area(
    "Enter English text",
    value=sample_text,
    height=160,
    placeholder="Type an English sentence here...",
)

translate_clicked = st.button("Translate", type="primary", use_container_width=True)

if translate_clicked:
    normalized_source = source_text.strip()
    if not normalized_source:
        st.warning("Please enter some English text first.")
    else:
        with st.spinner("Generating translation..."):
            translated_text_arr, time_taken = translate_text(
                normalized_source,
                model=model,
                tokenizer=tokenizer,
            )

        cleaned_translations = [text.strip() for text in translated_text_arr if text.strip()]
        st.session_state["translated_options"] = cleaned_translations
        st.session_state["translation_time_ms"] = time_taken * 1000
        st.session_state["last_source_text"] = normalized_source
        st.session_state["submitted_translation"] = None

translation_options = st.session_state.get("translated_options", [])
translation_time_ms = st.session_state.get("translation_time_ms")
last_source_text = st.session_state.get("last_source_text", "")

if translation_options:
    st.subheader("Hindi Translation Options")

    selected_index = st.radio(
        "Choose the statement you prefer",
        options=range(len(translation_options)),
        format_func=lambda idx: translation_options[idx],
        index=0,
        key="selected_hindi_option_idx",
    )
    preferred_translation = translation_options[selected_index]

    st.success(preferred_translation)

    submit_clicked = st.button("Submit Preferred Statement", use_container_width=True)
    if submit_clicked:
        st.session_state["submitted_translation"] = preferred_translation
        try:
            if not last_source_text:
                raise RuntimeError("No source text found. Please translate again before submitting.")

            inserted_id = save_translation_record(
                en_text=last_source_text,
                hindi_options=translation_options,
                selected=str(preferred_translation),
                selected_index=int(selected_index),
            )
            st.success(f"Selected translation saved to MongoDB. Document ID: {inserted_id}")
        except Exception as error:
            st.error(f"Unable to save translation: {error}")
            if "TLS handshake failed" in str(error):
                st.info(
                    "TLS fix tips: verify system date/time, then run `pip install -U certifi pymongo`. "
                    "If your network does SSL inspection, set `MONGO_TLS_ALLOW_INVALID_CERTS=true` in `.env` for local testing only."
                )

    submitted_translation = st.session_state.get("submitted_translation")
    if submitted_translation:
        st.info(f"Submitted statement: {submitted_translation}")

    if translation_time_ms is not None:
        st.caption(f"Time taken: {translation_time_ms:.2f} ms")
