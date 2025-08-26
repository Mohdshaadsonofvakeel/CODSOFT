
import os
import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Spam SMS Detector", page_icon="📱", layout="centered")

st.title("📱 Spam SMS Detector")
st.caption("TF-IDF + Classic ML (CodSoft Task 4)")

@st.cache_resource
def load_pipeline(path: str):
    return joblib.load(path)

# Sidebar: model selection
default_model_path = "spam_sms_naivebayes_tfidf_pipeline.joblib"
model_file = st.sidebar.text_input("Path to .joblib pipeline", value=default_model_path)
uploaded = st.sidebar.file_uploader("...or upload a .joblib file", type=["joblib"])

pipeline = None
if uploaded is not None:
    tmp_path = "uploaded_pipeline.joblib"
    with open(tmp_path, "wb") as f:
        f.write(uploaded.read())
    model_file = tmp_path

try:
    pipeline = load_pipeline(model_file)
    st.sidebar.success(f"Loaded: {model_file}")
except Exception as e:
    st.sidebar.warning("Could not load pipeline. Use the text box or upload a .joblib file.")

st.subheader("Try it out")
sms = st.text_area("Paste an SMS message here:", height=120, placeholder="e.g., Congratulations! You won a prize. Claim at http://bit.ly/...")

def predict_one(text: str):
    if pipeline is None:
        return None, None
    pred = pipeline.predict([text])[0]
    prob = None
    if hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba([text])[0]
            classes = list(getattr(pipeline, "classes_", []))
            if classes and "spam" in classes:
                prob = float(proba[classes.index("spam")])
            else:
                prob = float(np.max(proba))
        except Exception:
            prob = None
    return pred, prob

col1, col2 = st.columns([1,1])
with col1:
    if st.button("Predict"):
        if not sms.strip():
            st.info("Please paste a message first.")
        else:
            label, prob = predict_one(sms)
            if label is None:
                st.error("Pipeline not loaded.")
            else:
                st.markdown(f"**Prediction:** :{'green_circle' if label=='ham' else 'red_circle'}: **{label.upper()}**")
                if prob is not None:
                    st.write(f"Spam probability: **{prob:.3f}**")
with col2:
    st.write("Examples")
    samples = [
        "Congratulations! You have won a $1000 gift card. Click http://bit.ly/xyz to claim now!",
        "Are we still on for dinner tonight at 7?",
        "URGENT: Your account was locked due to suspicious activity. Verify at www.security-check.com",
        "hey, call me when you're free",
    ]
    if st.button("Run examples"):
        for s in samples:
            label, prob = predict_one(s)
            st.write(f"- **{label.upper() if label else 'N/A'}** — {s}")
            if prob is not None:
                st.caption(f"Spam probability: {prob:.3f}")
