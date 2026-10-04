import html

import joblib
import pandas as pd
import streamlit as st

from features import extract_features

st.set_page_config(page_title="PhishGuard", page_icon="🛡️", layout="centered")

st.markdown(
    """
<style>
.hero {
    padding: 2rem 1.5rem; border-radius: 20px; text-align: center;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
    color: white; margin-bottom: 1.5rem;
    box-shadow: 0 10px 30px rgba(124, 58, 237, 0.35);
}
.hero h1 { margin: 0; font-size: 2.4rem; color: white; }
.hero p { margin: .4rem 0 0; opacity: .9; font-size: 1.05rem; }

.result { padding: 1.4rem 1.6rem; border-radius: 16px; margin: 1rem 0; }
.result.bad  { background: rgba(239,68,68,.12);  border: 1px solid #ef4444; }
.result.good { background: rgba(34,197,94,.12);  border: 1px solid #22c55e; }
.result h2 { margin: 0 0 .3rem; }
.result .url { font-family: monospace; opacity: .8; word-break: break-all; font-size: .9rem; }

.meter { position: relative; height: 14px; border-radius: 99px; margin: 1.6rem 0 .4rem;
    background: linear-gradient(90deg, #22c55e, #eab308, #ef4444); }
.marker { position: absolute; top: -6px; width: 6px; height: 26px; border-radius: 4px;
    background: white; box-shadow: 0 0 8px rgba(0,0,0,.6); transform: translateX(-50%); }
.meter-labels { display: flex; justify-content: space-between; font-size: .8rem; opacity: .7; }

.flag { padding: .55rem .9rem; border-radius: 10px; margin: .35rem 0; font-size: .95rem; }
.flag.red   { background: rgba(239,68,68,.10); border-left: 4px solid #ef4444; }
.flag.green { background: rgba(34,197,94,.10); border-left: 4px solid #22c55e; }

div.stButton > button { border-radius: 12px; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    bundle = joblib.load("models/phishing_model.pkl")
    return bundle["model"], bundle["features"]


model, feature_names = load_model()

SAMPLES = {
    "✅ Google": "https://www.google.com",
    "✅ GitHub repo": "https://github.com/microsoft/vscode",
    "⚠️ Fake PayPal": "http://secure-login-paypal-verify.com/account/update",
    "⚠️ IP address": "http://192.168.1.5/login.php",
    "⚠️ Crypto bonus": "http://free-crypto-bonus-claim.top/wallet/confirm",
}


def explain(f: dict):
    """Turn raw features into plain-English reasons."""
    reds, greens = [], []
    if f["has_ip"]:
        reds.append("Uses a raw IP address instead of a domain name")
    if f["has_at"]:
        reds.append("Contains '@', which can hide the real destination")
    if not f["is_https"]:
        reds.append("Does not use HTTPS")
    else:
        greens.append("Uses HTTPS")
    if f["suspicious_word_count"] > 0:
        reds.append(f"Contains {f['suspicious_word_count']} suspicious keyword(s) such as login, verify or secure")
    if f["num_hyphens_domain"] >= 2:
        reds.append("Domain name has several hyphens")
    if f["num_subdomains"] >= 3:
        reds.append("Unusually many subdomains")
    if f["url_length"] > 75:
        reds.append(f"Very long URL ({f['url_length']} characters)")
    if f["domain_has_digits"]:
        reds.append("Domain name contains digits")
    if f["has_port"]:
        reds.append("Uses a non-standard port")
    if not reds:
        greens.append("No common phishing red flags found")
    if f["url_length"] <= 40 and not f["has_ip"]:
        greens.append("Short, simple URL")
    return reds, greens


if "history" not in st.session_state:
    st.session_state.history = []


def use_sample(u):
    st.session_state.url_input = u
    st.session_state.auto_check = True


with st.sidebar:
    st.header("🛡️ About")
    st.write(
        "PhishGuard looks at the **structure** of a URL (length, symbols, "
        "keywords, domain pattern) and a machine learning model estimates "
        "how likely it is to be phishing."
    )
    st.subheader("How it works")
    st.markdown(
        "1. You paste a URL\n"
        "2. 23 features are extracted\n"
        "3. A trained model gives a probability\n"
        "4. You see the verdict and the reasons"
    )
    st.info("It never opens the link, so checking is safe.")
    if st.button("🗑️ Clear history"):
        st.session_state.history = []

st.markdown(
    """
<div class="hero">
  <h1>🛡️ PhishGuard</h1>
  <p>Paste a link. Find out if it's a trap before you click.</p>
</div>
""",
    unsafe_allow_html=True,
)

url = st.text_input("Enter a URL", key="url_input", placeholder="https://example.com/login")

st.caption("Try an example:")
cols = st.columns(len(SAMPLES))
for col, (label, u) in zip(cols, SAMPLES.items()):
    col.button(label, on_click=use_sample, args=(u,), use_container_width=True)

check = st.button("🔍 Check URL", type="primary", use_container_width=True)
auto = st.session_state.pop("auto_check", False)

if check or auto:
    if not url.strip():
        st.warning("Please enter a URL.")
    else:
        feats_dict = extract_features(url)
        feats = pd.DataFrame([feats_dict])[feature_names]
        prob = float(model.predict_proba(feats)[0][1])
        is_bad = prob >= 0.5
        safe_url = html.escape(url.strip())

        if is_bad:
            st.markdown(
                f'<div class="result bad"><h2>🚨 Likely PHISHING</h2>'
                f'<div class="url">{safe_url}</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="result good"><h2>✅ Likely LEGITIMATE</h2>'
                f'<div class="url">{safe_url}</div></div>',
                unsafe_allow_html=True,
            )

        c1, c2, c3 = st.columns(3)
        c1.metric("Phishing probability", f"{prob:.1%}")
        c2.metric("Confidence", f"{max(prob, 1 - prob):.1%}")
        c3.metric("Risk level", "High" if prob >= 0.75 else "Medium" if prob >= 0.4 else "Low")

        st.markdown(
            f'<div class="meter"><div class="marker" style="left:{prob * 100:.1f}%"></div></div>'
            '<div class="meter-labels"><span>Safe</span><span>Suspicious</span><span>Dangerous</span></div>',
            unsafe_allow_html=True,
        )

        reds, greens = explain(feats_dict)
        st.subheader("Why this verdict?")
        for r in reds:
            st.markdown(f'<div class="flag red">🚩 {html.escape(r)}</div>', unsafe_allow_html=True)
        for g in greens:
            st.markdown(f'<div class="flag green">👍 {html.escape(g)}</div>', unsafe_allow_html=True)

        with st.expander("🔬 All extracted features"):
            st.dataframe(feats.T.rename(columns={0: "value"}), use_container_width=True)

        st.session_state.history.insert(
            0, {"URL": url.strip()[:70], "Verdict": "Phishing" if is_bad else "Legitimate",
                "Phishing %": round(prob * 100, 1)}
        )

if st.session_state.history:
    st.subheader("🕘 Recent checks")
    st.dataframe(pd.DataFrame(st.session_state.history[:10]), use_container_width=True, hide_index=True)

st.caption(
    "⚠️ This is a machine learning estimate, not a guarantee. "
    "Never enter passwords on a site you are unsure about."
)