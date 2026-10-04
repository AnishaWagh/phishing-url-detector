import joblib
import pandas as pd
import streamlit as st

from features import extract_features

st.set_page_config(page_title="Phishing URL Detector", page_icon="🛡️")


@st.cache_resource
def load_model():
    bundle = joblib.load("models/phishing_model.pkl")
    return bundle["model"], bundle["features"]


model, feature_names = load_model()

st.title("🛡️ Phishing URL Detector")
st.write("Paste a URL and the model will estimate whether it is phishing.")

url = st.text_input("URL", placeholder="https://example.com/login")

if st.button("Check URL", type="primary"):
    if not url.strip():
        st.warning("Please enter a URL.")
    else:
        feats = pd.DataFrame([extract_features(url)])[feature_names]
        phishing_prob = float(model.predict_proba(feats)[0][1])

        if phishing_prob >= 0.5:
            st.error(f"⚠️ Likely PHISHING ({phishing_prob:.1%} confidence)")
        else:
            st.success(f"✅ Likely LEGITIMATE ({1 - phishing_prob:.1%} confidence)")

        st.progress(phishing_prob, text=f"Phishing probability: {phishing_prob:.1%}")

        with st.expander("Extracted features"):
            st.dataframe(feats.T.rename(columns={0: "value"}))

st.caption(
    "This is a machine learning estimate, not a guarantee. "
    "Never enter credentials on a site you are unsure about."
)