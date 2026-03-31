from __future__ import annotations

import streamlit as st

from inference import device, load_model, translate_text


st.set_page_config(
    page_title="English to Hindi Translator",
    page_icon="🌐",
    layout="centered",
)


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
    if not source_text.strip():
        st.warning("Please enter some English text first.")
    else:
        with st.spinner("Generating translation..."):
            translated_text,time_taken = translate_text(
                source_text,
                model=model,
                tokenizer=tokenizer,
            )

        st.subheader("Hindi Translation")
        st.success(f"{translated_text or 'No translation was generated.'} \n\n⏱️ Time taken: {time_taken*1000:.2f}ms")
